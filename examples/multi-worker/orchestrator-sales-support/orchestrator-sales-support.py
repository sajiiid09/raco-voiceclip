#
# Copyright (c) 2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Jarvis-style Sales + Support orchestration demo using z.ai GLM.

This example keeps the user-facing voice pipeline in one main worker and
consults Sales and Support as background job workers. The main GLM speaks the
final response; the specialists return structured recommendations and progress
updates that are forwarded to the RTVI UI channel.
"""

import json
import os
import re
import time
from typing import Any

from dotenv import load_dotenv
from loguru import logger
from openai import AsyncOpenAI

from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.bus.ui import (
    BusUIJobCompletedMessage,
    BusUIJobGroupCompletedMessage,
    BusUIJobGroupStartedMessage,
    BusUIJobUpdateMessage,
)
from pipecat.pipeline.job_context import JobGroupError, JobStatus
from pipecat.pipeline.job_decorator import job
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.workers.base_worker import BaseWorker

load_dotenv(override=True)

MAIN_NAME = "acme-orchestrator"
SALES_NAME = "sales"
SUPPORT_NAME = "support"

ZAI_BASE_URL = os.environ.get("ZAI_BASE_URL", "https://open.bigmodel.cn/api/paas/v4")
ZAI_MODEL = os.environ.get("ZAI_MODEL", "glm-4")
CARTESIA_VOICE_ID = os.environ.get("CARTESIA_VOICE_ID", "9626c31c-bec5-4cca-baa8-f8ba9e84c8bc")

SPECIALIST_RESPONSE_SCHEMA = {
    "agent": "sales",
    "summary": "Short finding for the orchestrator",
    "confidence": 0.0,
    "recommended_action": "answer",
    "visible_feedback": "Short user-facing status for the UI",
}

SALES_SYSTEM_PROMPT = (
    "You are the Sales specialist for Acme Corp. You advise the main voice "
    "orchestrator. Focus only on product fit, pricing, availability, comparisons, "
    "and purchase intent. Products: Rocket Boots ($299, 60 mph, commuter and "
    "thrill use), Invisible Paint ($49/can, 24-hour invisibility, privacy and "
    "creative projects), Tornado Kit ($199, portable F1-F2 tornado generator). "
    "Return only compact JSON with keys: agent, summary, confidence, "
    "recommended_action, visible_feedback. recommended_action must be one of "
    "answer, ask_followup, route_to_other_agent, end."
)

SUPPORT_SYSTEM_PROMPT = (
    "You are the Support specialist for Acme Corp. You advise the main voice "
    "orchestrator. Focus only on troubleshooting, warranty, returns, setup, and "
    "technical how-to questions. Rocket Boots: recalibrate after 50 uses by "
    "holding the left heel reset for 5 seconds; adjust ankle strap balance dial "
    "for thrust imbalance; 1-year warranty. Invisible Paint: clean dry surface, "
    "two thin coats, visible under UV by design; no consumable warranty. Tornado "
    "Kit: check 4 AA batteries, red LED means low power, clean intake vents; "
    "2-year warranty. Return policy is 30 days unused, 5-7 business day refund. "
    "Return only compact JSON with keys: agent, summary, confidence, "
    "recommended_action, visible_feedback. recommended_action must be one of "
    "answer, ask_followup, route_to_other_agent, end."
)

MAIN_SYSTEM_PROMPT = (
    "You are Acme Orchestrator, a concise Jarvis-like voice agent. The user only "
    "hears you. You have two background specialists: Sales and Support. Use the "
    "consult_specialists tool whenever a specialist would improve the answer. "
    "Use Sales for product fit, pricing, availability, comparisons, and purchase "
    "intent. Use Support for troubleshooting, warranty, returns, setup, and "
    "technical how-to. Use both only when the user mixes sales and support needs "
    "or the intent is ambiguous. After tool results arrive, synthesize one brief "
    "spoken answer. Do not mention internal tool mechanics unless the user asks. "
    "Do not use markdown, bullets, emojis, or long lists."
)

SALES_KEYWORDS = {
    "buy",
    "price",
    "pricing",
    "cost",
    "purchase",
    "product",
    "compare",
    "recommend",
    "available",
    "availability",
}

SUPPORT_KEYWORDS = {
    "broken",
    "issue",
    "problem",
    "warranty",
    "return",
    "refund",
    "fix",
    "troubleshoot",
    "calibration",
    "calibrate",
    "battery",
    "charge",
    "setup",
    "not working",
}

VALID_ACTIONS = {"answer", "ask_followup", "route_to_other_agent", "end"}


def _now_ms() -> int:
    return int(time.time() * 1000)


def _zai_api_key() -> str:
    return os.environ.get("ZAI_API_KEY", os.environ.get("OPENAI_API_KEY", ""))


def _create_zai_llm(system_prompt: str):
    """Create a z.ai GLM LLM service through the OpenAI-compatible API."""
    from pipecat.services.openai.llm import OpenAILLMService

    return OpenAILLMService(
        api_key=_zai_api_key(),
        base_url=ZAI_BASE_URL,
        settings=OpenAILLMService.Settings(
            model=ZAI_MODEL,
            system_instruction=system_prompt,
            temperature=0.3,
        ),
    )


def select_specialists(query: str, routing: str = "auto") -> list[str]:
    """Select specialists for a user query.

    Args:
        query: User query or turn summary.
        routing: ``auto``, ``sales``, ``support``, or ``both``.

    Returns:
        Ordered specialist names.
    """
    requested = (routing or "auto").strip().lower()
    if requested in {"sales", SALES_NAME}:
        return [SALES_NAME]
    if requested in {"support", SUPPORT_NAME}:
        return [SUPPORT_NAME]
    if requested in {"both", "all", "sales_and_support"}:
        return [SALES_NAME, SUPPORT_NAME]

    text = query.lower()
    wants_sales = any(keyword in text for keyword in SALES_KEYWORDS)
    wants_support = any(keyword in text for keyword in SUPPORT_KEYWORDS)
    if wants_sales and wants_support:
        return [SALES_NAME, SUPPORT_NAME]
    if wants_support:
        return [SUPPORT_NAME]
    return [SALES_NAME]


def _extract_json_object(text: str) -> dict[str, Any]:
    if not text:
        return {}
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            return {}
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return {}


def normalize_specialist_response(agent: str, data: dict[str, Any]) -> dict[str, Any]:
    """Normalize specialist output into the demo's public response shape."""
    summary = str(data.get("summary") or "No clear specialist finding was produced.").strip()
    visible_feedback = str(data.get("visible_feedback") or summary).strip()
    action = str(data.get("recommended_action") or "answer").strip().lower()
    if action not in VALID_ACTIONS:
        action = "answer"
    try:
        confidence = float(data.get("confidence", 0.5))
    except (TypeError, ValueError):
        confidence = 0.5
    confidence = max(0.0, min(1.0, confidence))
    return {
        "agent": agent,
        "summary": summary[:500],
        "confidence": confidence,
        "recommended_action": action,
        "visible_feedback": visible_feedback[:240],
    }


def _required_env_missing() -> list[str]:
    names = ["ZAI_API_KEY", "DEEPGRAM_API_KEY", "CARTESIA_API_KEY", "DAILY_API_KEY"]
    missing = [name for name in names if not os.environ.get(name)]
    if "ZAI_API_KEY" in missing and os.environ.get("OPENAI_API_KEY"):
        missing.remove("ZAI_API_KEY")
    return missing


class SpecialistWorker(BaseWorker):
    """Background GLM worker that answers one specialist consultation job."""

    def __init__(self, name: str, *, system_prompt: str):
        super().__init__(name=name, active=True)
        self._system_prompt = system_prompt
        self._client = AsyncOpenAI(api_key=_zai_api_key() or "missing-key", base_url=ZAI_BASE_URL)

    @job(name="consult", sequential=True)
    async def consult(self, message):
        payload = message.payload or {}
        query = str(payload.get("query") or "")
        reason = str(payload.get("reason") or "")
        await self.send_job_update(
            message.job_id,
            {
                "phase": "thinking",
                "message": f"{self.name.title()} is reviewing the request.",
            },
        )
        await self.send_job_update(
            message.job_id,
            {
                "phase": "checking_context",
                "message": f"{self.name.title()} is checking Acme context.",
            },
        )

        try:
            raw = await self._complete(query=query, reason=reason)
            response = normalize_specialist_response(self.name, raw)
            await self.send_job_update(
                message.job_id,
                {
                    "phase": "forming_recommendation",
                    "message": response["visible_feedback"],
                },
            )
            await self.send_job_response(message.job_id, response=response)
        except Exception as e:
            logger.exception(f"Specialist '{self.name}' failed")
            await self.send_job_response(
                message.job_id,
                response=normalize_specialist_response(
                    self.name,
                    {
                        "summary": f"{self.name.title()} failed: {e}",
                        "confidence": 0.0,
                        "recommended_action": "route_to_other_agent",
                        "visible_feedback": f"{self.name.title()} could not complete the check.",
                    },
                ),
                status=JobStatus.FAILED,
            )

    async def _complete(self, *, query: str, reason: str) -> dict[str, Any]:
        user_prompt = (
            "User query:\n"
            f"{query}\n\n"
            "Reason the orchestrator requested you:\n"
            f"{reason or 'No extra routing reason supplied.'}\n\n"
            "Return JSON only. Example shape:\n"
            f"{json.dumps(SPECIALIST_RESPONSE_SCHEMA)}"
        )
        completion = await self._client.chat.completions.create(
            model=ZAI_MODEL,
            messages=[
                {"role": "system", "content": self._system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
        )
        content = completion.choices[0].message.content or ""
        return _extract_json_object(content)


def build_sales() -> SpecialistWorker:
    return SpecialistWorker(SALES_NAME, system_prompt=SALES_SYSTEM_PROMPT)


def build_support() -> SpecialistWorker:
    return SpecialistWorker(SUPPORT_NAME, system_prompt=SUPPORT_SYSTEM_PROMPT)


async def consult_specialists(
    params: Any,
    query: str,
    routing: str = "auto",
    reason: str = "",
) -> dict[str, Any]:
    """Consult Sales and/or Support specialists.

    Args:
        query: The user's current request, rewritten clearly for specialists.
        routing: One of ``auto``, ``sales``, ``support``, or ``both``.
        reason: Short reason why specialist help is needed.

    Returns:
        Structured specialist findings for the main orchestrator to synthesize.
    """
    worker = params.pipeline_worker
    specialists = select_specialists(query, routing)
    payload = {"query": query, "reason": reason, "routing": routing}
    label = "Consulting " + " + ".join(name.title() for name in specialists)

    try:
        async with worker.job_group(
            *specialists,
            name="consult",
            payload=payload,
            timeout=30.0,
            cancel_on_error=False,
        ) as group:
            await worker.send_bus_message(
                BusUIJobGroupStartedMessage(
                    source=worker.name,
                    target=None,
                    job_id=group.job_id,
                    workers=specialists,
                    label=label,
                    cancellable=False,
                    at=_now_ms(),
                )
            )
            async for event in group:
                await worker.send_bus_message(
                    BusUIJobUpdateMessage(
                        source=worker.name,
                        target=None,
                        job_id=group.job_id,
                        worker_name=event.worker_name,
                        data=event.data,
                        at=_now_ms(),
                    )
                )

            responses = group.responses
            for specialist_name in specialists:
                await worker.send_bus_message(
                    BusUIJobCompletedMessage(
                        source=worker.name,
                        target=None,
                        job_id=group.job_id,
                        worker_name=specialist_name,
                        status=str(JobStatus.COMPLETED),
                        response=responses.get(specialist_name, {}),
                        at=_now_ms(),
                    )
                )
            await worker.send_bus_message(
                BusUIJobGroupCompletedMessage(
                    source=worker.name,
                    target=None,
                    job_id=group.job_id,
                    at=_now_ms(),
                )
            )
            return {"selected_agents": specialists, "findings": responses}
    except JobGroupError as e:
        logger.warning(f"Specialist consultation failed: {e}")
        return {
            "selected_agents": specialists,
            "findings": {},
            "error": f"Specialist consultation failed: {e}",
        }


def _build_transport_params():
    from pipecat.transports.daily.transport import DailyParams

    return {
        "daily": lambda: DailyParams(audio_in_enabled=True, audio_out_enabled=True),
        "webrtc": lambda: TransportParams(audio_in_enabled=True, audio_out_enabled=True),
    }


async def run_bot(transport: BaseTransport, runner_args: Any):
    """Set up and run the orchestrated Sales + Support voice demo."""
    missing = _required_env_missing()
    if missing:
        raise RuntimeError(f"Missing required environment variables: {', '.join(missing)}")

    from pipecat.audio.vad.silero import SileroVADAnalyzer
    from pipecat.audio.vad.vad_analyzer import VADParams
    from pipecat.frames.frames import LLMRunFrame
    from pipecat.pipeline.pipeline import Pipeline
    from pipecat.pipeline.worker import PipelineParams, PipelineWorker
    from pipecat.services.cartesia.tts import CartesiaTTSService
    from pipecat.services.deepgram.stt import DeepgramSTTService
    from pipecat.workers.runner import WorkerRunner

    runner = WorkerRunner(handle_sigint=runner_args.handle_sigint)

    stt = DeepgramSTTService(api_key=os.environ["DEEPGRAM_API_KEY"])
    tts = CartesiaTTSService(
        api_key=os.environ["CARTESIA_API_KEY"],
        settings=CartesiaTTSService.Settings(voice=CARTESIA_VOICE_ID),
    )
    main_llm = _create_zai_llm(MAIN_SYSTEM_PROMPT)
    main_llm.register_direct_function(consult_specialists, timeout_secs=35.0)

    context = LLMContext(tools=ToolsSchema(standard_tools=[consult_specialists]))
    aggregators = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            vad_analyzer=SileroVADAnalyzer(
                params=VADParams(confidence=0.7, start_secs=0.2, stop_secs=0.5, min_volume=0.6)
            ),
        ),
    )

    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            aggregators.user(),
            main_llm,
            tts,
            transport.output(),
            aggregators.assistant(),
        ]
    )

    worker = PipelineWorker(
        pipeline,
        name=MAIN_NAME,
        params=PipelineParams(enable_metrics=True, enable_usage_metrics=True),
        idle_timeout_secs=runner_args.pipeline_idle_timeout_secs,
    )

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport_instance, client):
        logger.info("Client connected to orchestration demo")
        context.add_message(
            {
                "role": "developer",
                "content": (
                    "Welcome the user to Acme's voice concierge in one sentence, "
                    "then ask what they need help with."
                ),
            }
        )
        await worker.queue_frame(LLMRunFrame())

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport_instance, client):
        logger.info("Client disconnected from orchestration demo")
        await runner.cancel()

    await runner.add_workers(build_sales(), build_support(), worker)
    await runner.run()


async def bot(runner_args: Any):
    from pipecat.runner.utils import create_transport

    transport = await create_transport(runner_args, _build_transport_params())
    await run_bot(transport, runner_args)


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()

#
# Copyright (c) 2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Sales + Support multi-agent voice handoff with z.ai GLM.

Demonstrates a two-agent voice orchestration system where a single main worker
owns the transport (Daily), STT (Deepgram), and TTS (Cartesia). Two child
LLM workers — a Sales agent and a Support agent — each run their own LLM
pipeline and hand off control between each other via tool calls.

Only one child is active at a time. All audio flows through the main worker's
shared TTS, so the user hears a single consistent voice.

The LLM provider is z.ai GLM (OpenAI-compatible API). vLLM support is
prepared for local model deployment.

Requirements:

- ZAI_API_KEY (z.ai GLM API key)
- DEEPGRAM_API_KEY
- CARTESIA_API_KEY
- DAILY_API_KEY (for Daily transport)

Optional (for vLLM local models):

- VLLM_BASE_URL (e.g. http://localhost:8000/v1)
- VLLM_API_KEY (dummy key, e.g. "vllm")
- VLLM_MODEL (e.g. "Qwen/Qwen2.5-72B-Instruct")
"""

import os

from dotenv import load_dotenv
from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.bus import BusBridgeProcessor
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.llm_service import FunctionCallParams
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.daily.transport import DailyParams
from pipecat.workers.llm import LLMWorker, LLMWorkerActivationArgs, tool
from pipecat.workers.runner import WorkerRunner

load_dotenv(override=True)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MAIN_NAME = "acme-orchestrator"

# z.ai GLM configuration (OpenAI-compatible endpoint)
ZAI_API_KEY = os.environ.get("ZAI_API_KEY", os.environ.get("OPENAI_API_KEY", ""))
ZAI_BASE_URL = os.environ.get("ZAI_BASE_URL", "https://open.bigmodel.cn/api/paas/v4")
ZAI_MODEL = os.environ.get("ZAI_MODEL", "glm-4")

# vLLM configuration (optional, for local model support)
VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1")
VLLM_API_KEY = os.environ.get("VLLM_API_KEY", "vllm")
VLLM_MODEL = os.environ.get("VLLM_MODEL", "Qwen/Qwen2.5-72B-Instruct")

# Cartesia voice ID (Jacqueline — a warm, professional voice)
CARTESIA_VOICE_ID = os.environ.get("CARTESIA_VOICE_ID", "9626c31c-bec5-4cca-baa8-f8ba9e84c8bc")

# Transport params for different runner modes
transport_params = {
    "daily": lambda: DailyParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
    ),
    "webrtc": lambda: TransportParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
    ),
}

# ---------------------------------------------------------------------------
# Sales + Support Agent Prompts
# ---------------------------------------------------------------------------

SALES_SYSTEM_PROMPT = (
    "You are a friendly and knowledgeable Sales Agent for Acme Corp. "
    "Your responsibilities are:\n"
    "- Greet customers warmly and make them feel welcome\n"
    "- Present and explain available products and services\n"
    "- Answer pricing, availability, and comparison questions\n"
    "- Help customers choose the right product for their needs\n\n"
    "Available products:\n"
    "1. Acme Rocket Boots — Jet-powered boots, $299. Run up to 60 mph. "
    "Perfect for commuters and thrill-seekers.\n"
    "2. Acme Invisible Paint — Makes anything invisible for 24 hours, $49/can. "
    "Great for privacy and creative projects.\n"
    "3. Acme Tornado Kit — Portable tornado generator, $199, batteries included. "
    "Ideal for outdoor events and weather enthusiasts.\n\n"
    "IMPORTANT RULES:\n"
    "- If the customer has a technical question, a problem with an existing "
    "product, or needs troubleshooting, call the transfer_to_agent tool with "
    "agent 'support'. Do NOT attempt to answer technical questions yourself.\n"
    "- If the customer says goodbye or wants to end the conversation, call "
    "the end_conversation tool.\n"
    "- Do NOT mention the transfer process to the customer. Just transition "
    "naturally and seamlessly.\n"
    "- Keep responses brief and conversational — this is a voice conversation, "
    "not an email. Avoid long lists or walls of text."
)

SUPPORT_SYSTEM_PROMPT = (
    "You are a patient and thorough Technical Support Agent for Acme Corp. "
    "Your responsibilities are:\n"
    "- Troubleshoot product issues reported by customers\n"
    "- Answer detailed technical and how-to questions\n"
    "- Walk customers through solutions step by step\n"
    "- Provide warranty and return information when needed\n\n"
    "Product knowledge base:\n"
    "- Acme Rocket Boots: Max speed 60 mph. Battery life 2 hours. "
    "Charge time 4 hours. Common issues: calibration needed after 50 uses "
    "(press reset button on left heel for 5 seconds), thrust imbalance "
    "(adjust balance dial on ankle strap). Warranty: 1 year.\n"
    "- Acme Invisible Paint: Coverage 50 sq ft per can. Duration 24 hours. "
    "Apply 2 even coats. Common issues: patchy coverage (surface must be "
    "clean and dry, apply thinner coats), paint visible under UV light "
    "(this is expected behavior). Warranty: none (consumable product).\n"
    "- Acme Tornado Kit: Generates F1-F2 tornadoes. Range 100 yards. "
    "Runtime 30 minutes per battery set (4x AA). Common issues: won't start "
    "(ensure batteries are inserted correctly, red LED indicates low power), "
    "weak tornado (clean intake vents, check battery level). Warranty: 2 years.\n\n"
    "Return policy: 30 days, unused condition. Refund processed in 5-7 business days.\n\n"
    "IMPORTANT RULES:\n"
    "- If the customer wants to browse products, make a purchase, or asks "
    "sales-related questions, call the transfer_to_agent tool with agent "
    "'sales'.\n"
    "- If the customer says goodbye or wants to end the conversation, call "
    "the end_conversation tool.\n"
    "- Do NOT mention the transfer process to the customer. Just transition "
    "naturally and seamlessly.\n"
    "- Keep responses brief and conversational — this is a voice conversation, "
    "not an email. Avoid long lists or walls of text."
)


# ---------------------------------------------------------------------------
# LLM Builder Functions (Phase 2 + Phase 6)
# ---------------------------------------------------------------------------


def _create_zai_llm(system_prompt: str) -> OpenAILLMService:
    """Create an LLM service using z.ai GLM (OpenAI-compatible API).

    Uses the ``base_url`` parameter to point the OpenAI client at z.ai's
    endpoint. Falls back to standard OpenAI if ZAI_API_KEY is not set but
    OPENAI_API_KEY is.

    Args:
        system_prompt: The system instruction for the LLM.

    Returns:
        A configured OpenAILLMService targeting z.ai GLM.
    """
    return OpenAILLMService(
        api_key=ZAI_API_KEY,
        base_url=ZAI_BASE_URL,
        settings=OpenAILLMService.Settings(
            model=ZAI_MODEL,
            system_instruction=system_prompt,
        ),
    )


def _create_vllm_llm(system_prompt: str) -> OpenAILLMService:
    """Create an LLM service using a local vLLM server (OpenAI-compatible).

    For use with local models like Qwen, Llama, etc. The vLLM server must
    be running and serving the model at VLLM_BASE_URL.

    Args:
        system_prompt: The system instruction for the LLM.

    Returns:
        A configured OpenAILLMService targeting a vLLM server.
    """
    return OpenAILLMService(
        api_key=VLLM_API_KEY,
        base_url=VLLM_BASE_URL,
        settings=OpenAILLMService.Settings(
            model=VLLM_MODEL,
            system_instruction=system_prompt,
        ),
    )


def _create_llm(system_prompt: str, provider: str = "zai") -> OpenAILLMService:
    """Create an LLM service for the specified provider.

    Args:
        system_prompt: The system instruction for the LLM.
        provider: Provider to use — "zai" (default) or "vllm".

    Returns:
        A configured LLM service instance.

    Raises:
        ValueError: If provider is not recognized.
    """
    if provider == "vllm":
        return _create_vllm_llm(system_prompt)
    elif provider == "zai":
        return _create_zai_llm(system_prompt)
    else:
        raise ValueError(f"Unknown LLM provider: {provider!r}. Use 'zai' or 'vllm'.")


# ---------------------------------------------------------------------------
# Agent Worker (Phase 3)
# ---------------------------------------------------------------------------


class SalesSupportAgent(LLMWorker):
    """LLM-only child worker with transfer/end tools for Sales or Support.

    Receives user context from the main worker via the bus, runs its LLM,
    and ships generated text frames back through the bus. The main worker's
    TTS converts the text into audio for the user.

    Passing ``bridged=()`` tells :class:`PipelineWorker` to wrap the LLM
    pipeline with bus edge processors so frames flow between this worker
    and the main worker automatically.
    """

    @tool(cancel_on_interruption=False)
    async def transfer_to_agent(self, params: FunctionCallParams, agent: str, reason: str):
        """Transfer the user to another agent.

        The transfer completes even if the user interrupts
        (``cancel_on_interruption=False``) so the handoff is never lost.

        Args:
            agent: The agent to transfer to ('sales' or 'support').
            reason: Why the user is being transferred.
        """
        logger.info(f"Agent '{self.name}': transferring to '{agent}' ({reason})")
        await self.activate_worker(
            agent,
            args=LLMWorkerActivationArgs(
                messages=[{"role": "developer", "content": reason}],
            ),
            deactivate_self=True,
            result_callback=params.result_callback,
        )

    @tool
    async def end_conversation(self, params: FunctionCallParams, reason: str):
        """End the conversation when the user says goodbye.

        Args:
            reason: Why the conversation is ending.
        """
        logger.info(f"Agent '{self.name}': ending conversation ({reason})")
        await self.end(
            reason=reason,
            messages=[{"role": "developer", "content": reason}],
            result_callback=params.result_callback,
        )


# ---------------------------------------------------------------------------
# Agent Builder Functions
# ---------------------------------------------------------------------------


def build_sales(provider: str = "zai") -> SalesSupportAgent:
    """Build the Sales agent worker.

    Args:
        provider: LLM provider to use ("zai" or "vllm").

    Returns:
        A configured SalesSupportAgent with Sales system prompt.
    """
    llm = _create_llm(SALES_SYSTEM_PROMPT, provider=provider)
    return SalesSupportAgent("sales", llm=llm, bridged=())


def build_support(provider: str = "zai") -> SalesSupportAgent:
    """Build the Support agent worker.

    Args:
        provider: LLM provider to use ("zai" or "vllm").

    Returns:
        A configured SalesSupportAgent with Support system prompt.
    """
    llm = _create_llm(SUPPORT_SYSTEM_PROMPT, provider=provider)
    return SalesSupportAgent("support", llm=llm, bridged=())


# ---------------------------------------------------------------------------
# Main Bot Entry Point (Phase 4)
# ---------------------------------------------------------------------------


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
    """Set up and run the multi-agent Sales + Support voice bot.

    Creates the main worker pipeline (transport → STT → bus bridge → TTS →
    transport output) and registers two child LLM workers (sales, support)
    with the shared bus.

    Args:
        transport: The transport instance (Daily, WebRTC, etc.).
        runner_args: Arguments from the runner (signal handling, timeouts).
    """
    logger.info("Starting Sales + Support multi-agent voice bot")

    # Determine LLM provider from environment
    provider = os.environ.get("LLM_PROVIDER", "zai")
    logger.info(f"Using LLM provider: {provider}")

    # --- Worker Runner (owns bus + registry) ---
    runner = WorkerRunner(handle_sigint=runner_args.handle_sigint)

    # --- STT: Deepgram ---
    stt = DeepgramSTTService(api_key=os.environ["DEEPGRAM_API_KEY"])

    # --- TTS: Cartesia (shared voice) ---
    tts = CartesiaTTSService(
        api_key=os.environ["CARTESIA_API_KEY"],
        settings=CartesiaTTSService.Settings(
            voice=CARTESIA_VOICE_ID,
        ),
    )

    # --- LLM Context + Aggregators ---
    context = LLMContext()
    aggregators = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            vad_analyzer=SileroVADAnalyzer(
                params=VADParams(
                    confidence=0.7,
                    start_secs=0.2,
                    stop_secs=0.5,
                    min_volume=0.6,
                ),
            ),
        ),
    )

    # --- Bus Bridge (mid-pipeline bidirectional bridge to bus) ---
    bridge = BusBridgeProcessor(
        bus=runner.bus,
        worker_name=MAIN_NAME,
        name=f"{MAIN_NAME}::BusBridge",
    )

    # --- Main Pipeline ---
    # transport.input → STT → UserAgg → BusBridge → TTS → transport.output → AssistantAgg
    #
    # User audio flows DOWNSTREAM from transport to BusBridge, where it's
    # forwarded to the active child worker via the bus.
    #
    # Child LLM text flows from the bus back through BusBridge (UPSTREAM),
    # then through TTS for audio synthesis, then to transport output.
    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            aggregators.user(),
            bridge,
            tts,
            transport.output(),
            aggregators.assistant(),
        ]
    )

    # --- Main Worker ---
    worker = PipelineWorker(
        pipeline,
        name=MAIN_NAME,
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
        idle_timeout_secs=runner_args.pipeline_idle_timeout_secs,
    )

    # --- Transport Event Handlers ---

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport_instance, client):
        """Activate the Sales agent when a user connects."""
        logger.info("Client connected — activating Sales agent")
        await worker.activate_worker(
            "sales",
            args=LLMWorkerActivationArgs(
                messages=[
                    {
                        "role": "developer",
                        "content": (
                            "A customer has just connected. Welcome them to "
                            "Acme Corp, briefly mention the available products "
                            "(Rocket Boots, Invisible Paint, Tornado Kit), and "
                            "ask how you can help them today."
                        ),
                    },
                ],
            ),
        )

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport_instance, client):
        """Gracefully shut down when the user disconnects."""
        logger.info("Client disconnected — shutting down")
        await runner.cancel()

    # --- Error Handler (Phase 6: Production Hardening) ---

    @worker.event_handler("on_pipeline_error")
    async def on_pipeline_error(worker_instance, error):
        """Handle pipeline errors gracefully.

        Non-fatal errors are logged but do not stop the pipeline.
        Fatal errors trigger cancellation.
        """
        logger.error(f"Pipeline error in main worker: {error}")

    # --- Register Workers and Run ---
    await runner.add_workers(
        build_sales(provider=provider),
        build_support(provider=provider),
        worker,
    )

    await runner.run()


async def bot(runner_args: RunnerArguments):
    """Main bot entry point compatible with Pipecat Cloud / development runner."""
    transport = await create_transport(runner_args, transport_params)
    await run_bot(transport, runner_args)


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()

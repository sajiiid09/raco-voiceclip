#
# Copyright (c) 2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for the Sales + Support orchestration demo."""

import asyncio
import importlib.util
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from pipecat.bus import (
    AsyncQueueBus,
    BusJobRequestMessage,
    BusJobResponseMessage,
    BusJobUpdateMessage,
)
from pipecat.bus.ui import (
    BusUIJobCompletedMessage,
    BusUIJobGroupCompletedMessage,
    BusUIJobGroupStartedMessage,
    BusUIJobUpdateMessage,
)
from pipecat.registry import WorkerRegistry
from pipecat.utils.asyncio.task_manager import TaskManager, TaskManagerParams

_BOT_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "examples",
    "multi-worker",
    "orchestrator-sales-support",
    "orchestrator-sales-support.py",
)


def _load_bot_module():
    spec = importlib.util.spec_from_file_location("orchestrator_sales_support", _BOT_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


async def _create_bus():
    bus = AsyncQueueBus()
    tm = TaskManager()
    tm.setup(TaskManagerParams(loop=asyncio.get_running_loop()))
    await bus.setup(tm)
    return bus, tm


class TestRoutingAndParsing(unittest.TestCase):
    def setUp(self):
        self.bot = _load_bot_module()

    def test_select_specialists_sales_only(self):
        selected = self.bot.select_specialists("How much do Rocket Boots cost?")
        self.assertEqual(selected, ["sales"])

    def test_select_specialists_support_only(self):
        selected = self.bot.select_specialists("My Tornado Kit battery is not working")
        self.assertEqual(selected, ["support"])

    def test_select_specialists_both_for_mixed_intent(self):
        selected = self.bot.select_specialists(
            "I want to buy Rocket Boots, but what is the warranty if they break?"
        )
        self.assertEqual(selected, ["sales", "support"])

    def test_normalize_specialist_response_clamps_and_defaults(self):
        normalized = self.bot.normalize_specialist_response(
            "sales",
            {
                "summary": "Useful answer",
                "confidence": 4,
                "recommended_action": "unknown",
            },
        )
        self.assertEqual(normalized["agent"], "sales")
        self.assertEqual(normalized["confidence"], 1.0)
        self.assertEqual(normalized["recommended_action"], "answer")
        self.assertEqual(normalized["visible_feedback"], "Useful answer")

    @patch.dict(
        os.environ,
        {
            "ZAI_API_KEY": "test-zai-key",
            "ZAI_BASE_URL": "https://test.invalid/v4",
            "ZAI_MODEL": "glm-test",
        },
    )
    def test_zai_llm_created_with_openai_compatible_config(self):
        pytest.importorskip("websockets")
        bot = _load_bot_module()
        llm = bot._create_zai_llm("System prompt")
        self.assertEqual(llm._settings.model, "glm-test")
        self.assertEqual(llm._settings.system_instruction, "System prompt")

    @patch.dict(os.environ, {}, clear=True)
    def test_required_env_reports_missing_provider_values(self):
        bot = _load_bot_module()
        self.assertEqual(
            bot._required_env_missing(),
            ["ZAI_API_KEY", "DEEPGRAM_API_KEY", "CARTESIA_API_KEY", "DAILY_API_KEY"],
        )


class TestSpecialistWorker(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.bot = _load_bot_module()
        self.bus, self.tm = await _create_bus()
        self.sent = []
        original_send = self.bus.send

        async def capture_send(message):
            self.sent.append(message)
            await original_send(message)

        self.bus.send = capture_send

    async def asyncTearDown(self):
        await self.bus.stop()

    async def test_specialist_job_emits_updates_and_response(self):
        worker = self.bot.SpecialistWorker("sales", system_prompt="Prompt")
        worker._complete = unittest.mock.AsyncMock(
            return_value={
                "agent": "sales",
                "summary": "Rocket Boots match commuter needs.",
                "confidence": 0.82,
                "recommended_action": "answer",
                "visible_feedback": "Sales found a strong product fit.",
            }
        )
        await worker.attach(registry=WorkerRegistry(runner_name="test"), bus=self.bus)
        request = BusJobRequestMessage(
            source="main",
            target="sales",
            job_id="job-1",
            job_name="consult",
            payload={"query": "Should I buy Rocket Boots?", "reason": "sales fit"},
        )
        worker._active_jobs[request.job_id] = request

        await worker.consult(request)

        updates = [m for m in self.sent if isinstance(m, BusJobUpdateMessage)]
        responses = [m for m in self.sent if isinstance(m, BusJobResponseMessage)]
        self.assertGreaterEqual(len(updates), 3)
        self.assertEqual(len(responses), 1)
        self.assertEqual(responses[0].response["agent"], "sales")
        self.assertEqual(responses[0].response["recommended_action"], "answer")


class _FakeJobGroup:
    job_id = "ui-job-1"
    responses = {
        "sales": {
            "agent": "sales",
            "summary": "Sales fit confirmed.",
            "confidence": 0.9,
            "recommended_action": "answer",
            "visible_feedback": "Sales finished.",
        }
    }

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    def __aiter__(self):
        self._events = iter(
            [
                SimpleNamespace(
                    worker_name="sales",
                    data={"phase": "thinking", "message": "Sales is thinking."},
                )
            ]
        )
        return self

    async def __anext__(self):
        try:
            return next(self._events)
        except StopIteration:
            raise StopAsyncIteration


class _FakePipelineWorker:
    name = "acme-orchestrator"

    def __init__(self):
        self.sent = []
        self.requested = None

    def job_group(self, *workers, **kwargs):
        self.requested = {"workers": workers, "kwargs": kwargs}
        return _FakeJobGroup()

    async def send_bus_message(self, message):
        self.sent.append(message)


class TestConsultSpecialistsTool(unittest.IsolatedAsyncioTestCase):
    async def test_consult_specialists_forwards_ui_lifecycle(self):
        bot = _load_bot_module()
        fake_worker = _FakePipelineWorker()
        params = SimpleNamespace(pipeline_worker=fake_worker)

        result = await bot.consult_specialists(
            params,
            query="How much do Rocket Boots cost?",
            routing="sales",
            reason="Pricing question",
        )

        self.assertEqual(fake_worker.requested["workers"], ("sales",))
        self.assertEqual(result["selected_agents"], ["sales"])
        self.assertTrue(any(isinstance(m, BusUIJobGroupStartedMessage) for m in fake_worker.sent))
        self.assertTrue(any(isinstance(m, BusUIJobUpdateMessage) for m in fake_worker.sent))
        self.assertTrue(any(isinstance(m, BusUIJobCompletedMessage) for m in fake_worker.sent))
        self.assertTrue(any(isinstance(m, BusUIJobGroupCompletedMessage) for m in fake_worker.sent))


if __name__ == "__main__":
    unittest.main()

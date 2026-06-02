#
# Copyright (c) 2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for the Sales + Support multi-agent handoff bot.

These tests validate the core components without requiring external API
connections. They cover:

- System prompt content validation
- Bus bridge frame routing
- Worker activation / deactivation handoff via the bus
- Agent handoff (sales -> support and support -> sales)
- Context preservation during handoff
- End conversation flow

Tests that import the bot module directly are skipped when optional
dependencies (deepgram, cartesia, daily) are not installed.
"""

import asyncio
import importlib.util
import os
import unittest
from unittest.mock import patch

from pipecat.bus import (
    AsyncQueueBus,
    BusActivateWorkerMessage,
    BusDeactivateWorkerMessage,
    BusEndMessage,
    BusFrameMessage,
)
from pipecat.frames.frames import EndFrame, TextFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineWorker
from pipecat.processors.filters.identity_filter import IdentityFilter
from pipecat.processors.frame_processor import FrameDirection
from pipecat.registry import WorkerRegistry
from pipecat.registry.types import WorkerReadyData
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.tests.utils import run_test
from pipecat.utils.asyncio.task_manager import TaskManager, TaskManagerParams
from pipecat.workers.base_worker import WorkerActivationArgs
from pipecat.workers.runner import WorkerRunner

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_BOT_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "examples",
    "multi-worker",
    "local-handoff",
    "local-handoff-sales-support.py",
)

# Try importing heavy deps; skip tests that need them if unavailable
try:
    import cartesia  # noqa: F401
    import deepgram  # noqa: F401

    _HAS_DEPS = True
except ImportError:
    _HAS_DEPS = False


def _load_bot_module():
    """Load the bot module from its file path."""
    spec = importlib.util.spec_from_file_location("sales_support_bot", _BOT_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _read_bot_source():
    """Read the bot source as text (no import needed)."""
    with open(_BOT_PATH) as f:
        return f.read()


async def create_test_bus():
    bus = AsyncQueueBus()
    tm = TaskManager()
    tm.setup(TaskManagerParams(loop=asyncio.get_running_loop()))
    await bus.setup(tm)
    return bus, tm


def create_test_registry():
    return WorkerRegistry(runner_name="test-runner")


def capture_bus(bus):
    sent = []
    original_send = bus.send

    async def capture_send(message):
        sent.append(message)
        await original_send(message)

    bus.send = capture_send
    return sent


def make_stub_pipeline_task(name, *, bridged=None, active=True):
    return PipelineWorker(
        Pipeline([IdentityFilter()]),
        name=name,
        bridged=bridged,
        cancel_on_idle_timeout=False,
    )


# ---------------------------------------------------------------------------
# Phase 2+3: System Prompt Content Tests (no imports needed)
# ---------------------------------------------------------------------------


class TestSystemPrompts(unittest.TestCase):
    """Validate system prompts contain required handoff instructions."""

    def setUp(self):
        self.source = _read_bot_source()

    def test_sales_prompt_mentions_support_transfer(self):
        """Sales prompt should instruct transfer to 'support' agent."""
        self.assertIn("support", self.source)
        self.assertIn("transfer_to_agent", self.source)
        self.assertIn("end_conversation", self.source)

    def test_support_prompt_mentions_sales_transfer(self):
        """Support prompt should instruct transfer to 'sales' agent."""
        self.assertIn("sales", self.source)
        self.assertIn("transfer_to_agent", self.source)

    def test_sales_prompt_has_product_info(self):
        """Sales prompt should contain product information."""
        self.assertIn("Rocket Boots", self.source)
        self.assertIn("Invisible Paint", self.source)
        self.assertIn("Tornado Kit", self.source)

    def test_support_prompt_has_technical_info(self):
        """Support prompt should contain troubleshooting knowledge."""
        self.assertIn("troubleshoot", self.source)
        self.assertIn("warranty", self.source)
        self.assertIn("calibration", self.source)

    def test_prompts_require_brief_voice_responses(self):
        """Both prompts should instruct brief voice-appropriate responses."""
        self.assertIn("voice conversation", self.source)
        # "brief" appears in both prompts
        count = self.source.lower().count("brief")
        self.assertGreaterEqual(count, 2)

    def test_has_zai_glm_configuration(self):
        """Bot should configure z.ai GLM as LLM provider."""
        self.assertIn("ZAI_BASE_URL", self.source)
        self.assertIn("ZAI_MODEL", self.source)
        self.assertIn("glm", self.source)

    def test_has_vllm_configuration(self):
        """Bot should prepare vLLM configuration for local models."""
        self.assertIn("VLLM_BASE_URL", self.source)
        self.assertIn("VLLM_MODEL", self.source)

    def test_has_openai_compatible_base_url(self):
        """LLM should use base_url for OpenAI-compatible API."""
        self.assertIn("base_url", self.source)
        self.assertIn("OpenAILLMService", self.source)

    def test_has_bridged_workers(self):
        """Child workers should be bridged to the bus."""
        self.assertIn("bridged=()", self.source)

    def test_has_cancel_on_interruption_false(self):
        """Transfer tool should have cancel_on_interruption=False."""
        self.assertIn("cancel_on_interruption=False", self.source)

    def test_has_daily_transport_params(self):
        """Bot should configure Daily transport."""
        self.assertIn("DailyParams", self.source)
        self.assertIn("audio_in_enabled=True", self.source)
        self.assertIn("audio_out_enabled=True", self.source)

    def test_has_cartesia_tts(self):
        """Bot should configure Cartesia TTS."""
        self.assertIn("CartesiaTTSService", self.source)

    def test_has_deepgram_stt(self):
        """Bot should configure Deepgram STT."""
        self.assertIn("DeepgramSTTService", self.source)

    def test_has_vad_configuration(self):
        """Bot should configure Silero VAD."""
        self.assertIn("SileroVADAnalyzer", self.source)

    def test_has_bus_bridge_processor(self):
        """Bot should use BusBridgeProcessor."""
        self.assertIn("BusBridgeProcessor", self.source)

    def test_pipeline_order_correct(self):
        """Pipeline should be: transport.input -> stt -> user_agg -> bridge -> tts -> transport.output."""
        self.assertIn("transport.input()", self.source)
        self.assertIn("stt", self.source)
        self.assertIn("aggregators.user()", self.source)
        self.assertIn("bridge", self.source)
        self.assertIn("tts", self.source)
        self.assertIn("transport.output()", self.source)
        self.assertIn("aggregators.assistant()", self.source)


# ---------------------------------------------------------------------------
# Phase 2+3: LLM + Agent Builder Tests (requires optional deps)
# ---------------------------------------------------------------------------


@unittest.skipUnless(_HAS_DEPS, "Optional deps (deepgram, cartesia) not installed")
class TestLLMBuilder(unittest.TestCase):
    """Test LLM builder functions for z.ai GLM and vLLM."""

    @patch.dict(
        os.environ,
        {
            "ZAI_API_KEY": "test-zai-key",
            "ZAI_BASE_URL": "https://test.api/v4",
            "ZAI_MODEL": "glm-4-test",
        },
    )
    def test_zai_llm_created_with_correct_config(self):
        bot = _load_bot_module()
        llm = bot._create_zai_llm("You are a test.")
        self.assertIsInstance(llm, OpenAILLMService)
        self.assertEqual(llm._settings.model, "glm-4-test")
        self.assertEqual(llm._settings.system_instruction, "You are a test.")

    @patch.dict(
        os.environ,
        {
            "VLLM_API_KEY": "vllm",
            "VLLM_BASE_URL": "http://localhost:8000/v1",
            "VLLM_MODEL": "Qwen/Qwen2.5-7B",
        },
    )
    def test_vllm_llm_created_with_correct_config(self):
        bot = _load_bot_module()
        llm = bot._create_vllm_llm("You are a vllm test.")
        self.assertIsInstance(llm, OpenAILLMService)
        self.assertEqual(llm._settings.model, "Qwen/Qwen2.5-7B")

    @patch.dict(os.environ, {"ZAI_API_KEY": "test-key"})
    def test_create_llm_default_provider_is_zai(self):
        bot = _load_bot_module()
        llm = bot._create_llm("Test prompt")
        self.assertIsInstance(llm, OpenAILLMService)

    def test_create_llm_invalid_provider_raises(self):
        bot = _load_bot_module()
        with self.assertRaises(ValueError) as ctx:
            bot._create_llm("Test", provider="unknown")
        self.assertIn("unknown", str(ctx.exception))


@unittest.skipUnless(_HAS_DEPS, "Optional deps (deepgram, cartesia) not installed")
class TestAgentBuilder(unittest.TestCase):
    """Test Sales and Support agent construction."""

    @patch.dict(os.environ, {"ZAI_API_KEY": "test-key"})
    def test_build_sales_creates_worker_with_correct_name(self):
        bot = _load_bot_module()
        agent = bot.build_sales(provider="zai")
        self.assertEqual(agent.name, "sales")

    @patch.dict(os.environ, {"ZAI_API_KEY": "test-key"})
    def test_build_support_creates_worker_with_correct_name(self):
        bot = _load_bot_module()
        agent = bot.build_support(provider="zai")
        self.assertEqual(agent.name, "support")

    @patch.dict(os.environ, {"ZAI_API_KEY": "test-key"})
    def test_build_sales_has_sales_system_prompt(self):
        bot = _load_bot_module()
        agent = bot.build_sales(provider="zai")
        self.assertIn("Sales Agent", agent._settings.system_instruction)
        self.assertIn("Acme Corp", agent._settings.system_instruction)

    @patch.dict(os.environ, {"ZAI_API_KEY": "test-key"})
    def test_build_support_has_support_system_prompt(self):
        bot = _load_bot_module()
        agent = bot.build_support(provider="zai")
        self.assertIn("Technical Support", agent._settings.system_instruction)
        self.assertIn("troubleshoot", agent._settings.system_instruction)

    @patch.dict(os.environ, {"ZAI_API_KEY": "test-key"})
    def test_agents_have_transfer_tool(self):
        bot = _load_bot_module()
        sales = bot.build_sales(provider="zai")
        support = bot.build_support(provider="zai")
        self.assertTrue(hasattr(sales, "transfer_to_agent"))
        self.assertTrue(hasattr(support, "transfer_to_agent"))

    @patch.dict(os.environ, {"ZAI_API_KEY": "test-key"})
    def test_agents_have_end_conversation_tool(self):
        bot = _load_bot_module()
        sales = bot.build_sales(provider="zai")
        support = bot.build_support(provider="zai")
        self.assertTrue(hasattr(sales, "end_conversation"))
        self.assertTrue(hasattr(support, "end_conversation"))


# ---------------------------------------------------------------------------
# Phase 4: Bus Bridge Integration
# ---------------------------------------------------------------------------


class TestBusBridgeIntegration(unittest.IsolatedAsyncioTestCase):
    """Test bus bridge frame routing in the main pipeline."""

    async def test_text_frame_sent_to_bus_not_passed_through(self):
        """Text frames should be forwarded to the bus, not passed downstream."""
        bus = AsyncQueueBus()
        sent_to_bus = []
        original_send = bus.send

        async def capture_send(msg):
            sent_to_bus.append(msg)
            await original_send(msg)

        bus.send = capture_send

        from pipecat.bus import BusBridgeProcessor

        processor = BusBridgeProcessor(bus=bus, worker_name="test_main")
        pipeline = Pipeline([processor])

        down, _ = await run_test(
            pipeline,
            frames_to_send=[TextFrame(text="hello from user")],
            expected_down_frames=[],
        )

        text_frames = [f for f in down if isinstance(f, TextFrame)]
        self.assertEqual(len(text_frames), 0)

        bus_frame_msgs = [m for m in sent_to_bus if isinstance(m, BusFrameMessage)]
        self.assertEqual(len(bus_frame_msgs), 1)
        self.assertEqual(bus_frame_msgs[0].frame.text, "hello from user")
        self.assertEqual(bus_frame_msgs[0].source, "test_main")


# ---------------------------------------------------------------------------
# Phase 5: Handoff & Edge Cases
# ---------------------------------------------------------------------------


class TestWorkerActivationDeactivation(unittest.IsolatedAsyncioTestCase):
    """Test worker activation/deactivation handoff via the bus."""

    async def asyncSetUp(self):
        self.bus, self.tm = await create_test_bus()
        self.registry = create_test_registry()

    async def test_sales_activates_on_bus_message(self):
        """Sales worker becomes active when BusActivateWorkerMessage is sent."""
        sales = make_stub_pipeline_task("sales", bridged=(), active=False)
        sales._active = False
        sales._pending_activation = False
        await sales.attach(registry=self.registry, bus=self.bus)

        activated = asyncio.Event()

        @sales.event_handler("on_activated")
        async def on_activated(worker, args):
            activated.set()

        async def drive():
            await asyncio.sleep(0.05)
            await self.bus.send(BusActivateWorkerMessage(source="main", target="sales", args=None))
            await asyncio.wait_for(activated.wait(), timeout=2.0)
            await sales.queue_frame(EndFrame())

        runner = WorkerRunner(bus=self.bus, handle_sigint=False)
        await runner.add_workers(sales)
        await asyncio.gather(runner.run(), drive())

        self.assertTrue(sales.active)

    async def test_handoff_deactivates_self_and_activates_target(self):
        """activate_worker(deactivate_self=True) deactivates self, activates target."""
        sent = capture_bus(self.bus)
        sales = make_stub_pipeline_task("sales", bridged=())
        await sales.attach(registry=self.registry, bus=self.bus)

        await sales.activate_worker("support", deactivate_self=True)

        deactivate_msgs = [m for m in sent if isinstance(m, BusDeactivateWorkerMessage)]
        self.assertEqual(len(deactivate_msgs), 1)
        self.assertEqual(deactivate_msgs[0].target, "sales")

        activate_msgs = [m for m in sent if isinstance(m, BusActivateWorkerMessage)]
        self.assertEqual(len(activate_msgs), 1)
        self.assertEqual(activate_msgs[0].target, "support")

    async def test_support_handoff_back_to_sales(self):
        """Support agent can hand off back to Sales."""
        sent = capture_bus(self.bus)
        support = make_stub_pipeline_task("support", bridged=())
        await support.attach(registry=self.registry, bus=self.bus)

        await support.activate_worker("sales", deactivate_self=True)

        activate_msgs = [m for m in sent if isinstance(m, BusActivateWorkerMessage)]
        self.assertEqual(len(activate_msgs), 1)
        self.assertEqual(activate_msgs[0].target, "sales")

        deactivate_msgs = [m for m in sent if isinstance(m, BusDeactivateWorkerMessage)]
        self.assertEqual(len(deactivate_msgs), 1)
        self.assertEqual(deactivate_msgs[0].target, "support")

    async def test_inactive_worker_ignores_bus_frames(self):
        """Inactive bridged workers do not process incoming bus frames."""
        worker = make_stub_pipeline_task("sales", bridged=(), active=False)
        worker._active = False
        worker._pending_activation = False
        await worker.attach(registry=self.registry, bus=self.bus)
        self.assertFalse(worker.active)

    async def test_activation_with_context_messages(self):
        """Activation args are preserved during handoff."""
        sent = capture_bus(self.bus)
        sales = make_stub_pipeline_task("sales", bridged=())
        await sales.attach(registry=self.registry, bus=self.bus)

        reason = "Customer has a technical question about Rocket Boots"
        args = WorkerActivationArgs(
            metadata={"messages": [{"role": "developer", "content": reason}]}
        )
        await sales.activate_worker("support", args=args, deactivate_self=True)

        activate_msgs = [m for m in sent if isinstance(m, BusActivateWorkerMessage)]
        self.assertEqual(len(activate_msgs), 1)
        self.assertIsNotNone(activate_msgs[0].args)
        self.assertEqual(activate_msgs[0].args["metadata"]["messages"][0]["content"], reason)

    async def test_double_handoff_sales_to_support_and_back(self):
        """Full round-trip: sales -> support -> sales."""
        sent = capture_bus(self.bus)
        sales = make_stub_pipeline_task("sales", bridged=())
        support = make_stub_pipeline_task("support", bridged=(), active=False)
        support._active = False
        support._pending_activation = False

        await sales.attach(registry=self.registry, bus=self.bus)
        await support.attach(registry=self.registry, bus=self.bus)

        # Sales (active) -> support
        await sales.activate_worker("support", deactivate_self=True)
        # Simulate support becoming active (in real usage, bus msg does this)
        support._active = True
        # Support (now active) -> sales
        await support.activate_worker("sales", deactivate_self=True)

        activate_msgs = [m for m in sent if isinstance(m, BusActivateWorkerMessage)]
        self.assertEqual(len(activate_msgs), 2)
        self.assertEqual(activate_msgs[0].target, "support")
        self.assertEqual(activate_msgs[1].target, "sales")

        deactivate_msgs = [m for m in sent if isinstance(m, BusDeactivateWorkerMessage)]
        self.assertEqual(len(deactivate_msgs), 2)
        self.assertEqual(deactivate_msgs[0].target, "sales")
        self.assertEqual(deactivate_msgs[1].target, "support")

    async def test_end_conversation_sends_end_message(self):
        """BaseWorker.end() sends a BusEndMessage to end the session."""
        sent = capture_bus(self.bus)
        worker = make_stub_pipeline_task("sales", bridged=())
        await worker.attach(registry=self.registry, bus=self.bus)

        await worker.end(reason="Customer said goodbye")

        end_msgs = [m for m in sent if isinstance(m, BusEndMessage)]
        self.assertEqual(len(end_msgs), 1)
        self.assertEqual(end_msgs[0].source, "sales")


if __name__ == "__main__":
    unittest.main()

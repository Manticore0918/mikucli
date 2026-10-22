from __future__ import annotations

import asyncio
import json
import sys
import tempfile
import time
import unittest
from pathlib import Path
from queue import Queue
from threading import Event
from types import SimpleNamespace
from unittest.mock import Mock

import httpx

from mikucli.cli_support.interactive import ApprovalBroker
from mikucli.http_requests import ProviderRequestError
from mikucli.langgraph_runtime import GraphAgentSession
from mikucli.llm import AssistantMessage, ModelClient, TokenUsage, ToolCall
from mikucli.mcp_runtime.server import ServerRuntime
from mikucli.memory import MemoryEntry, MemoryType
from mikucli.multi_agent import OrchestratorSession
from mikucli.react import AgentSession
from mikucli.run_control import RunLimitExceeded, RunLimits
from mikucli.tools import ToolRegistry, ToolResult
from mikucli.workspace import Workspace


def message(content="done", calls=None, tokens=3):
    return AssistantMessage(
        content=content,
        tool_calls=calls or [],
        raw={},
        token_usage=TokenUsage(
            prompt_tokens=2, completion_tokens=tokens - 2, total_tokens=tokens
        ),
    )


class QuietConsole:
    def __getattr__(self, name):
        return lambda *args, **kwargs: None


class SequenceClient:
    def __init__(self, *responses):
        self.responses = iter(responses)
        self.calls = 0

    def chat(self, **kwargs):
        self.calls += 1
        value = next(self.responses)
        if isinstance(value, Exception):
            raise value
        return value


class TaskOutcomeTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def session(self, session_type, client, **kwargs):
        return session_type(
            client=client,
            model="test",
            workspace=self.root,
            tools=kwargs.pop("tools", ToolRegistry(Workspace(self.root))),
            console=QuietConsole(),
            **kwargs,
        )

    def test_success_step_limit_cancellation_and_provider_failure_in_both_runtimes(
        self,
    ):
        for runtime in (AgentSession, GraphAgentSession):
            for responses, kwargs, expected in (
                ([message()], {}, ("succeeded", "completed", 0)),
                (
                    [message("", [ToolCall("call", "list_files", {})])],
                    {"max_steps": 1},
                    ("limit_reached", "max_steps", 3),
                ),
                (
                    [],
                    {"stop_requested": lambda: True},
                    ("stopped", "user_request", 130),
                ),
                ([ProviderRequestError("outage")], {}, ("failed", "provider_error", 1)),
            ):
                with self.subTest(runtime=runtime, expected=expected):
                    stop = kwargs.pop("stop_requested", None)
                    result = self.session(
                        runtime, SequenceClient(*responses), **kwargs
                    ).run_turn("task", stop_requested=stop)
                    self.assertEqual(
                        (result.status, result.reason, result.exit_code), expected
                    )
                    log = json.loads(result.log_path.read_text(encoding="utf-8"))
                    self.assertEqual(log["metadata"]["outcome"]["status"], expected[0])
                    self.assertEqual(log["events"][-1]["type"], "task_finished")

    def test_unresolved_denial_fails_without_mutation(self):
        client = SequenceClient(
            message(
                "",
                [
                    ToolCall(
                        "call", "write_file", {"path": "denied.txt", "content": "x"}
                    )
                ],
            ),
            message("Blocked."),
        )
        result = self.session(AgentSession, client).run_turn("write")
        self.assertEqual((result.status, result.reason), ("failed", "approval_denied"))
        self.assertFalse((self.root / "denied.txt").exists())

    def test_graph_can_run_beyond_default_graph_recursion_limit(self):
        client = SequenceClient(
            *[message("", [ToolCall(f"c{i}", "list_files", {})]) for i in range(20)],
            message(),
        )
        result = self.session(
            GraphAgentSession, client, max_steps=25, memory_window_entries=100
        ).run_turn("task")
        self.assertEqual(result.status, "succeeded")
        self.assertEqual(client.calls, 21)

    def test_token_limit_and_missing_usage_prevent_following_tool_execution(self):
        for response, reason in (
            (
                message(
                    "",
                    [
                        ToolCall(
                            "call", "write_file", {"path": "late.txt", "content": "x"}
                        )
                    ],
                    tokens=5,
                ),
                "token_limit",
            ),
            (AssistantMessage("done", [], {}, TokenUsage()), "usage_unavailable"),
        ):
            with self.subTest(reason=reason):
                result = self.session(
                    AgentSession,
                    SequenceClient(response),
                    run_limits=RunLimits(max_tokens=5),
                ).run_turn("task")
                self.assertEqual(
                    (result.status, result.reason), ("limit_reached", reason)
                )
                self.assertFalse((self.root / "late.txt").exists())

    def test_spend_is_accumulated_and_turn_budget_is_reset(self):
        client = SequenceClient(message(), message())
        session = self.session(
            AgentSession,
            client,
            run_limits=RunLimits(
                max_spend=0.000002,
                prompt_price_per_million=1,
                completion_price_per_million=1,
            ),
        )
        for _ in range(2):
            result = session.run_turn("task")
            self.assertEqual(result.reason, "spend_limit")
            self.assertAlmostEqual(result.metrics["estimated_spend"], 0.000003)

    def test_spend_overflow_fails_closed_with_valid_json_metrics(self):
        response = message(tokens=10_000_000)
        result = self.session(
            AgentSession,
            SequenceClient(response),
            run_limits=RunLimits(
                max_spend=1,
                prompt_price_per_million=1e308,
                completion_price_per_million=1e308,
            ),
        ).run_turn("task")
        self.assertEqual(result.reason, "usage_unavailable")
        self.assertIsNone(result.metrics["estimated_spend"])
        json.dumps(result.to_dict(), allow_nan=False)

    def test_task_deadline_cancels_inflight_model_request(self):
        cancelled = Event()

        async def handler(request):
            try:
                await asyncio.sleep(5)
            except asyncio.CancelledError:
                cancelled.set()
                raise
            return httpx.Response(
                200, json={"choices": [{"message": {"content": "late"}}]}
            )

        for runtime in (AgentSession, GraphAgentSession):
            cancelled.clear()
            client = ModelClient(
                "dummy",
                "https://example.test/chat",
                transport=httpx.MockTransport(handler),
            )
            started = time.monotonic()
            result = self.session(
                runtime, client, run_limits=RunLimits(max_duration_seconds=0.1)
            ).run_turn("task")
            self.assertEqual(result.reason, "duration_limit")
            self.assertTrue(cancelled.is_set())
            self.assertLess(time.monotonic() - started, 2)

    def test_shell_output_limit_terminates_process_and_never_calls_model_again(self):
        tools = ToolRegistry(Workspace(self.root), confirm_tool=lambda _: True)
        command = f'"{sys.executable}" -c "import sys; sys.stdout.write(\'x\'*2000000); sys.stdout.flush()"'
        client = SequenceClient(
            message(
                "",
                [ToolCall("call", "run_shell", {"command": command, "reason": "test"})],
            )
        )
        result = self.session(
            AgentSession,
            client,
            tools=tools,
            run_limits=RunLimits(max_output_bytes=128),
        ).run_turn("task")
        self.assertEqual(
            (result.status, result.reason), ("limit_reached", "output_limit")
        )
        self.assertEqual(tools._active_processes, set())
        self.assertEqual(client.calls, 1)

    def test_tool_result_is_clipped_before_persistence_and_display(self):
        (self.root / "large.txt").write_text("x" * 200, encoding="utf-8")
        client = SequenceClient(
            message("", [ToolCall("call", "read_file", {"path": "large.txt"})])
        )
        result = self.session(
            AgentSession, client, run_limits=RunLimits(max_output_bytes=32)
        ).run_turn("task")
        self.assertEqual(result.reason, "output_limit")
        log = json.loads(result.log_path.read_text(encoding="utf-8"))
        output = next(
            event["content"]
            for event in log["events"]
            if event["type"] == "tool_result"
        )
        self.assertEqual(len(output.encode("utf-8")), 32)

    def test_team_propagates_planner_and_worker_failure(self):
        planner_limit = SequenceClient(
            message("", [ToolCall("call", "list_files", {})])
        )
        result = self.session(OrchestratorSession, planner_limit, max_steps=1).run_turn(
            "task"
        )
        self.assertEqual((result.status, result.reason), ("limit_reached", "max_steps"))
        plan = message(json.dumps({"steps": [{"id": "one", "task": "work"}]}))
        worker_failure = SequenceClient(plan, ProviderRequestError("outage"))
        result = self.session(OrchestratorSession, worker_failure).run_turn("task")
        self.assertEqual((result.status, result.reason), ("failed", "provider_error"))
        self.assertEqual(worker_failure.calls, 2)

    def test_team_token_budget_is_shared_across_planner_and_workers(self):
        plan = message(
            json.dumps(
                {
                    "steps": [
                        {"id": "one", "task": "work"},
                        {"id": "two", "task": "work"},
                    ]
                }
            ),
            tokens=5,
        )
        client = SequenceClient(plan, message(tokens=4))
        result = self.session(
            OrchestratorSession, client, run_limits=RunLimits(max_tokens=8)
        ).run_turn("task")
        self.assertEqual(
            (result.status, result.reason), ("limit_reached", "token_limit")
        )
        self.assertEqual(result.metrics["reported_tokens"], 9)
        self.assertEqual(client.calls, 2)

    def test_compression_model_calls_count_toward_the_same_token_limit(self):
        client = SequenceClient(
            message("", [ToolCall("call", "list_files", {})]),
            message("summary"),
            message("[]"),
        )
        session = self.session(
            AgentSession,
            client,
            context_window_tokens=2,
            run_limits=RunLimits(max_tokens=9),
            retain_recent_rounds=0,
            memory_retriever=SimpleNamespace(retrieve=lambda **_: []),
        )
        session.memory.old_entries.append(
            MemoryEntry(MemoryType.CONVERSATION, [], content="old context")
        )
        result = session.run_turn("task")
        self.assertEqual(result.reason, "token_limit")
        self.assertEqual(result.metrics["reported_tokens"], 9)
        self.assertEqual(client.calls, 3)
        self.assertTrue(session.memory.old_entries)

    def test_duration_limit_releases_pending_approval_without_mutating(self):
        broker = ApprovalBroker()
        tools = ToolRegistry(Workspace(self.root), confirm_tool=broker.request)
        client = SequenceClient(
            message(
                "",
                [ToolCall("call", "write_file", {"path": "late.txt", "content": "x"})],
            )
        )
        result = self.session(
            AgentSession,
            client,
            tools=tools,
            run_limits=RunLimits(max_duration_seconds=0.1),
        ).run_turn("task")
        self.assertEqual(result.reason, "duration_limit")
        self.assertFalse(broker.has_pending())
        self.assertFalse((self.root / "late.txt").exists())

    def test_duration_limit_terminates_running_shell(self):
        tools = ToolRegistry(Workspace(self.root), confirm_tool=lambda _: True)
        command = f'"{sys.executable}" -c "import time; time.sleep(5)"'
        client = SequenceClient(
            message(
                "",
                [ToolCall("call", "run_shell", {"command": command, "reason": "test"})],
            )
        )
        result = self.session(
            AgentSession,
            client,
            tools=tools,
            run_limits=RunLimits(max_duration_seconds=0.1),
        ).run_turn("task")
        self.assertEqual(result.reason, "duration_limit")
        self.assertEqual(tools._active_processes, set())

    def test_mcp_deadline_cancels_connection_and_guards_queued_command(self):
        runtime = object.__new__(ServerRuntime)
        runtime._closed = False
        runtime._commands = Queue()
        runtime._cancel_main_task = Mock()
        runtime.close = Mock()
        tools = SimpleNamespace(
            schemas=list, invoke=lambda *_: runtime._submit("call_tool")
        )
        client = SequenceClient(message("", [ToolCall("call", "remote_tool", {})]))
        result = self.session(
            AgentSession,
            client,
            tools=tools,
            run_limits=RunLimits(max_duration_seconds=0.1),
        ).run_turn("task")
        self.assertEqual(result.reason, "duration_limit")
        runtime._cancel_main_task.assert_called_once()
        runtime.close.assert_called_once()
        command = runtime._commands.get_nowait()
        with self.assertRaises(RunLimitExceeded):
            command.guard()

    def test_cancelled_multi_tool_response_keeps_history_valid_for_next_prompt(self):
        stop = Event()
        invoked = []

        def invoke(name, arguments):
            invoked.append(name)
            stop.set()
            return ToolResult(ok=True, content="first tool completed")

        tools = SimpleNamespace(schemas=list, invoke=invoke)
        for runtime in (AgentSession, GraphAgentSession):
            with self.subTest(runtime=runtime):
                stop.clear()
                invoked.clear()
                client = SequenceClient(
                    message(
                        "",
                        [ToolCall("first", "one", {}), ToolCall("second", "two", {})],
                    ),
                    message("next prompt works"),
                )
                session = self.session(
                    runtime, client, tools=tools, memory_window_entries=100
                )
                stopped = session.run_turn("task", stop_requested=stop.is_set)
                self.assertEqual(stopped.status, "stopped")
                replies = [
                    item for item in session.memory.messages() if item["role"] == "tool"
                ]
                self.assertEqual(
                    [item["tool_call_id"] for item in replies], ["first", "second"]
                )
                self.assertIn("not executed", replies[1]["content"])
                self.assertEqual(invoked, ["one"])
                stop.clear()
                self.assertEqual(session.run_turn("next").status, "succeeded")


if __name__ == "__main__":
    unittest.main()

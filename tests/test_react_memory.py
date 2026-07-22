from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

from mikucli.llm import AssistantMessage, TokenUsage, ToolCall
from mikucli.memory import LongTermMemory, MemoryEntry, MemoryType
from mikucli.observability.recorder import LocalTraceRecorder
from mikucli.observability.store import LocalTraceStore
from mikucli.react import AgentSession
from mikucli.skills import Skill, SkillScope
from mikucli.tools import ToolRegistry
from mikucli.workspace import Workspace


class FakeClient:
    def __init__(self, responses: list[AssistantMessage]) -> None:
        self.responses = responses
        self.requests: list[list[dict[str, Any]]] = []

    def chat(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        stream: bool = False,
    ) -> AssistantMessage:
        self.requests.append(messages)
        return self.responses.pop(0)


class FakeConsole:
    def __init__(self) -> None:
        self.progress_messages: list[str] = []

    def progress(self, message: str) -> None:
        self.progress_messages.append(message)

    def tool_request(self, name: str, arguments: dict[str, Any]) -> None:
        pass

    def tool_result(self, name: str, ok: bool, content: str, diff: str = "") -> None:
        pass

    def answer(self, content: str) -> None:
        pass

    def token_usage(self, usage: TokenUsage) -> None:
        pass


class AgentSessionMemoryTests(unittest.TestCase):
    def test_stop_request_prevents_the_next_model_call(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            client = FakeClient([])
            session = AgentSession(
                client=client,  # type: ignore[arg-type]
                model="test-model",
                workspace=root,
                tools=ToolRegistry(Workspace(root)),
                console=FakeConsole(),
            )

            result = session.run_turn("long task", stop_requested=lambda: True)

            self.assertEqual(result.final_answer, "Stopped by user.")
            self.assertEqual(client.requests, [])

    def test_active_skill_is_ephemeral_and_recorded_in_run_log(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            trace_store = LocalTraceStore(root / ".mikucli" / "observability")
            client = FakeClient(
                [
                    AssistantMessage(content="first done", tool_calls=[], raw={}, token_usage=TokenUsage(total_tokens=1)),
                    AssistantMessage(content="second done", tool_calls=[], raw={}, token_usage=TokenUsage(total_tokens=1)),
                ]
            )
            session = AgentSession(
                client=client,  # type: ignore[arg-type]
                model="test-model",
                workspace=root,
                tools=ToolRegistry(Workspace(root)),
                console=FakeConsole(),
                trace_recorder=LocalTraceRecorder(trace_store, stale_after_seconds=None),
            )
            skill = Skill(
                name="review",
                description="Review code.",
                instructions="Use the active review workflow.",
                scope=SkillScope.WORKSPACE,
                path=root / ".mikucli" / "skills" / "review" / "SKILL.md",
                content_hash="a" * 64,
                metadata={"name": "review", "description": "Review code."},
            )

            first = session.run_turn("inspect auth", active_skill=skill)
            payload = json.loads(first.log_path.read_text(encoding="utf-8"))
            session.run_turn("follow up")

            self.assertIn("Active Skill: $review", client.requests[0][0]["content"])
            self.assertIn("Use the active review workflow.", client.requests[0][0]["content"])
            self.assertNotIn("Active Skill: $review", client.requests[1][0]["content"])
            self.assertEqual(payload["task_prompt"], "inspect auth")
            self.assertEqual(
                payload["metadata"]["skill"],
                {"name": "review", "scope": "workspace", "content_hash": "a" * 64},
            )
            trace = trace_store.fetch_one(
                "select attributes_json from traces where trace_id = ?",
                (payload["metadata"]["trace_id"],),
            )
            self.assertIsNotNone(trace)
            trace_attributes = json.loads(trace["attributes_json"] if trace is not None else "{}")
            self.assertEqual(trace_attributes["skill.name"], "review")
            self.assertEqual(trace_attributes["skill.scope"], "workspace")
            self.assertEqual(trace_attributes["skill.content_hash"], "a" * 64)

    def test_compresses_old_memory_when_token_usage_exceeds_threshold(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "README.md").write_text("hello", encoding="utf-8")
            console = FakeConsole()
            client = FakeClient(
                [
                    AssistantMessage(
                        content="",
                        tool_calls=[ToolCall(id="call_1", name="list_files", arguments={})],
                        raw={},
                        token_usage=TokenUsage(total_tokens=50),
                    ),
                    AssistantMessage(
                        content="done",
                        tool_calls=[],
                        raw={},
                        token_usage=TokenUsage(total_tokens=81),
                    ),
                    AssistantMessage(
                        content="compressed old context",
                        tool_calls=[],
                        raw={},
                        token_usage=TokenUsage(total_tokens=10),
                    ),
                    AssistantMessage(
                        content='["User prefers concise answers."]',
                        tool_calls=[],
                        raw={},
                        token_usage=TokenUsage(total_tokens=10),
                    ),
                ]
            )
            workspace = Workspace(root)
            long_term_memory = LongTermMemory(root / ".mikucli" / "long_term_memory.json")
            session = AgentSession(
                client=client,  # type: ignore[arg-type]
                model="test-model",
                workspace=root,
                tools=ToolRegistry(workspace),
                console=console,
                context_window_tokens=100,
                memory_window_entries=1,
                long_term_memory=long_term_memory,
                retain_recent_rounds=0,
            )
            session.memory.old_entries.append(
                MemoryEntry(
                    type=MemoryType.CONVERSATION,
                    messages=[{"role": "user", "content": "old context"}],
                    content="old context",
                )
            )

            result = session.run_turn("list files")

            self.assertEqual(result.final_answer, "done")
            self.assertEqual(session.memory.old_entries, [])
            self.assertEqual(len(session.memory.summary_entries), 1)
            self.assertTrue(any("Compressed" in message for message in console.progress_messages))
            self.assertIn("Session memory summary", session.memory.summary_entries[0].messages[0]["content"])
            self.assertEqual([record.content for record in long_term_memory.records], ["User prefers concise answers."])


if __name__ == "__main__":
    unittest.main()

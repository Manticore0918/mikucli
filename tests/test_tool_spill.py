from __future__ import annotations

import json
import re
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from types import SimpleNamespace

from mikucli.langgraph_runtime import GraphAgentSession
from mikucli.llm import AssistantMessage, TokenUsage, ToolCall
from mikucli.memory import SessionMemory
from mikucli.memory.__main__ import main as memory_main
from mikucli.memory.archive import SessionArchive
from mikucli.memory.compression import MapReduceContextCompressor
from mikucli.memory.tool_spill import (
    DEFAULT_MAX_TOOL_INLINE_BYTES,
    effective_tool_inline_limit,
    spill_tool_output,
)
from mikucli.react import AgentSession
from mikucli.tools import ToolResult


def _silent_console() -> SimpleNamespace:
    return SimpleNamespace(
        **{
            name: lambda *a, **k: None
            for name in (
                "progress",
                "answer",
                "token_usage",
                "tool_request",
                "tool_result",
            )
        }
    )


class GiantOutputTools:
    def __init__(self, payload: str, *, ok: bool = True) -> None:
        self.payload = payload
        self.ok = ok

    def schemas(self) -> list[dict]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "giant_tool",
                    "description": "Return a large payload.",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]

    def invoke(self, name: str, arguments: dict) -> ToolResult:
        if name == "giant_tool":
            return ToolResult(ok=self.ok, content=self.payload)
        return ToolResult(ok=False, content=f"unexpected tool: {name}")

    def read_only_tool_names(self) -> set[str]:
        return {"giant_tool"}


class ToolSpillTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.archive = SessionArchive(self.root / ".mikucli" / "session_memory.sqlite3")

    def _memory(self) -> SessionMemory:
        return SessionMemory(
            {"role": "system", "content": "system"},
            archive=self.archive,
            retriever=SimpleNamespace(retrieve=lambda **_: []),
        )

    def test_exact_threshold_stays_inline_and_over_threshold_spills_with_head_tail(self) -> None:
        memory = self._memory()
        exact = "a" * 2048
        kept = spill_tool_output(
            memory=memory,
            tool_name="run_shell",
            tool_call_id="c1",
            ok=True,
            content=exact,
            max_inline_bytes=2048,
        )
        self.assertFalse(kept.spilled)
        self.assertEqual(kept.content, exact)
        self.assertEqual(kept.archived_source_id, "")

        # Multi-byte UTF-8 + newlines exceeding threshold spills cleanly.
        lines = [f"HEAD_LINE_{i:04d} 日志 🧩" for i in range(40)]
        lines.append("MIDDLE_SECRET_TOKEN_987654321")
        lines.extend(f"TAIL_LINE_{i:04d} 结束 ✅" for i in range(40))
        large = "\n".join(lines)
        spilled = spill_tool_output(
            memory=memory,
            tool_name="run_shell",
            tool_call_id="c2",
            ok=True,
            content=large,
            max_inline_bytes=2048,
        )
        self.assertTrue(spilled.spilled)
        self.assertTrue(spilled.archived_source_id)
        self.assertLessEqual(len(spilled.content.encode("utf-8")), 2048)
        self.assertIn("HEAD_LINE_0000", spilled.content)
        self.assertIn("TAIL_LINE_0039", spilled.content)
        self.assertNotIn("MIDDLE_SECRET_TOKEN_987654321", spilled.content)
        self.assertIn(spilled.archived_source_id, spilled.content)
        self.assertIn("recall_session_source", spilled.content)

        # Full exact original output is archived with valid checksum.
        recovered = memory.get_source(spilled.archived_source_id)
        self.assertEqual(recovered.content, large)
        self.assertEqual(recovered.messages, [])

        # Offline CLI can read the exact spilled source without credentials.
        output = StringIO()
        with redirect_stdout(output):
            code = memory_main(
                [
                    "source",
                    "--workspace",
                    str(self.root),
                    "--scope-id",
                    memory.scope_id,
                    "--entry-id",
                    spilled.archived_source_id,
                ]
            )
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(output.getvalue())["content"], large)

    def test_effective_limit_clamps_to_small_context_window(self) -> None:
        self.assertEqual(
            effective_tool_inline_limit(DEFAULT_MAX_TOOL_INLINE_BYTES, 1_000_000),
            DEFAULT_MAX_TOOL_INLINE_BYTES,
        )
        # For a 16,000-token hard window: output_reserve=2000 -> (16000-2000)//4 = 3500.
        self.assertEqual(
            effective_tool_inline_limit(DEFAULT_MAX_TOOL_INLINE_BYTES, 16_000),
            3500,
        )

    def test_recall_session_source_is_exempt_from_spilling(self) -> None:
        memory = self._memory()
        payload = json.dumps({"content": "中" * 4000}, ensure_ascii=False)
        result = spill_tool_output(
            memory=memory,
            tool_name="recall_session_source",
            tool_call_id="c-recall",
            ok=True,
            content=payload,
            max_inline_bytes=1024,
        )
        self.assertFalse(result.spilled)
        self.assertEqual(result.content, payload)

    def test_corrupt_archive_still_bounds_preview_and_reports_unavailability(self) -> None:
        memory = self._memory()
        self.archive.path.parent.mkdir(parents=True, exist_ok=True)
        self.archive.path.write_bytes(b"not a sqlite database")
        huge = "BEGIN\n" + ("payload-line\n" * 500) + "END_MARKER"
        spilled = spill_tool_output(
            memory=memory,
            tool_name="run_shell",
            tool_call_id="c-corrupt",
            ok=True,
            content=huge,
            max_inline_bytes=2048,
        )
        self.assertTrue(spilled.spilled)
        self.assertEqual(spilled.archived_source_id, "")
        self.assertTrue(spilled.archive_error)
        self.assertIn("archive_status: unavailable", spilled.content)
        self.assertIn("BEGIN", spilled.content)
        self.assertIn("END_MARKER", spilled.content)
        self.assertLessEqual(len(spilled.content.encode("utf-8")), 2048)

    def test_oversized_tool_output_never_blows_hardcap_and_is_recallable_in_both_runtimes(
        self,
    ) -> None:
        # 120KB payload > 40,000 hardcap! Without spill-to-archive this would fail with context_window_exceeded.
        middle_needle = "EXACT_MIDDLE_NEEDLE_XYZ_777"
        giant_payload = (
            "HEADER_START\n"
            + ("filler-head-line\n" * 3500)
            + f"{middle_needle}\n"
            + ("filler-tail-line\n" * 3500)
            + "FOOTER_TRACEBACK_END"
        )
        self.assertGreater(len(giant_payload.encode("utf-8")), 100_000)

        for runtime in (AgentSession, GraphAgentSession):
            for native in (True, False):
                with self.subTest(runtime=runtime.__name__, native=native):
                    requests: list[list[dict]] = []
                    recalled_chunks: list[str] = []

                    def chat(
                        *,
                        messages,
                        requests=requests,
                        recalled_chunks=recalled_chunks,
                        native=native,
                        **kwargs,
                    ):
                        requests.append(messages)
                        step = len(requests)
                        if step == 1:
                            if native:
                                return AssistantMessage(
                                    "",
                                    [ToolCall("call-giant", "giant_tool", {})],
                                    {},
                                    TokenUsage(total_tokens=10),
                                )
                            return AssistantMessage(
                                json.dumps({"tool": "giant_tool", "arguments": {}}),
                                [],
                                {},
                                TokenUsage(total_tokens=10),
                            )
                        if step == 2:
                            tool_msg = messages[-1]["content"]
                            self.assertIn("HEADER_START", tool_msg)
                            self.assertIn("FOOTER_TRACEBACK_END", tool_msg)
                            self.assertNotIn(middle_needle, tool_msg)
                            match = re.search(r"archived_source_id:\s*([0-9a-f]+)", tool_msg)
                            self.assertIsNotNone(match, tool_msg)
                            source_id = match.group(1)
                            # Account for JSON escaping (\n -> \\n) in recall_session_source output.
                            escaped_offset = len(json.dumps(giant_payload[: giant_payload.index(middle_needle)]))
                            args = {
                                "source_id": source_id,
                                "offset": max(0, escaped_offset - 100),
                                "limit": 2000,
                            }
                            if native:
                                return AssistantMessage(
                                    "",
                                    [ToolCall("call-recall", "recall_session_source", args)],
                                    {},
                                    TokenUsage(total_tokens=10),
                                )
                            return AssistantMessage(
                                json.dumps(
                                    {"tool": "recall_session_source", "arguments": args}
                                ),
                                [],
                                {},
                                TokenUsage(total_tokens=10),
                            )
                        recalled_chunks.append(messages[-1]["content"])
                        return AssistantMessage("recovered", [], {}, TokenUsage(total_tokens=10))

                    session = runtime(
                        client=SimpleNamespace(chat=chat),
                        model="test",
                        workspace=self.root,
                        tools=GiantOutputTools(giant_payload),
                        console=_silent_console(),
                        context_window_tokens=40_000,
                        context_soft_cap_tokens=20_000,
                        max_tool_inline_bytes=4_096,
                        memory_retriever=SimpleNamespace(retrieve=lambda **_: []),
                    )
                    result = session.run_turn("Run giant tool and recover middle needle")
                    self.assertEqual(result.status, "succeeded", result.final_answer)
                    self.assertEqual(result.final_answer, "recovered")
                    self.assertEqual(len(requests), 3)
                    self.assertIn(middle_needle, recalled_chunks[0])

                    # Run log records spill metadata.
                    log_payload = json.loads(result.log_path.read_text(encoding="utf-8"))
                    giant_events = [
                        event
                        for event in log_payload["events"]
                        if event["type"] == "tool_result" and event["tool"] == "giant_tool"
                    ]
                    self.assertEqual(len(giant_events), 1)
                    self.assertTrue(giant_events[0]["output_spilled"])
                    self.assertTrue(giant_events[0]["archived_source_id"])

                    # Subsequent compression commits preview entry cleanly without source ID collision.
                    session.memory.prepare_entries_for_compression(0)

                    def compress_chat(*, messages, **kwargs):
                        content = messages[-1]["content"]
                        if "covered_parts" in content:
                            parts = json.loads(content.split("\n\n", 1)[1])
                            return AssistantMessage(
                                json.dumps(
                                    {
                                        "summary": "compressed spilled turn",
                                        "covered_parts": [part["part_id"] for part in parts],
                                        "records": [],
                                        "durable_facts": [],
                                    }
                                ),
                                [],
                                {},
                                TokenUsage(total_tokens=5),
                            )
                        return AssistantMessage("merged", [], {}, TokenUsage(total_tokens=5))

                    compressor = MapReduceContextCompressor(
                        client=SimpleNamespace(chat=compress_chat), model="test"
                    )
                    summary = session.memory.compress_old_entries(compressor)
                    self.assertIsNotNone(summary)

    def test_failed_spilled_tool_captures_bounded_blocker_and_spilled_evidence_orders_after_it(
        self,
    ) -> None:
        fail_payload = "ERROR_HEAD: build failed\n" + ("trace line\n" * 1000) + "ERROR_TAIL"
        requests: list[list[dict]] = []

        def chat(*, messages, **kwargs):
            requests.append(messages)
            if len(requests) == 1:
                return AssistantMessage(
                    "",
                    [ToolCall("call-fail", "giant_tool", {})],
                    {},
                    TokenUsage(total_tokens=10),
                )
            return AssistantMessage("saw failure", [], {}, TokenUsage(total_tokens=10))

        session = AgentSession(
            client=SimpleNamespace(chat=chat),
            model="test",
            workspace=self.root,
            tools=GiantOutputTools(fail_payload, ok=False),
            console=_silent_console(),
            context_window_tokens=40_000,
            max_tool_inline_bytes=4_096,
            memory_retriever=SimpleNamespace(retrieve=lambda **_: []),
        )
        result = session.run_turn("Never change `src/api.py`.")
        self.assertEqual(result.status, "succeeded", result.final_answer)
        blockers = [r for r in session.memory.ledger.snapshot() if r.kind == "blocker"]
        self.assertEqual(len(blockers), 1)
        self.assertLessEqual(len(blockers[0].quote), 1000)
        self.assertIn("archived_source_id:", blockers[0].quote)

        # A later spilled tool output can be cited to resolve the blocker, but NOT a user constraint.
        later_source_id = session.memory.archive_tool_output(
            tool_name="run_shell",
            tool_call_id="call-fix",
            ok=True,
            content="BUILD_FIXED_OK\n" + ("log\n" * 500),
        )
        session.memory.update_task_state(
            blockers[0].record_id, "completed", later_source_id, "BUILD_FIXED_OK"
        )
        self.assertEqual(
            session.memory.ledger.records[blockers[0].record_id].status, "completed"
        )
        constraint = next(
            r for r in session.memory.ledger.snapshot() if r.kind == "constraint"
        )
        with self.assertRaises(ValueError):
            session.memory.update_task_state(
                constraint.record_id, "superseded", later_source_id, "BUILD_FIXED_OK"
            )


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from mikucli.memory import (
    LongTermMemory,
    LongTermMemoryRecord,
    MapReduceContextCompressor,
    MemoryEntry,
    MemoryRetriever,
    MemoryType,
    SessionMemory,
    default_long_term_memory_path,
    token_usage_ratio,
)
from mikucli.llm import AssistantMessage, TokenUsage


class StaticCompressor:
    def compress(self, entries: list[MemoryEntry]) -> MemoryEntry | None:
        if not entries:
            return None
        return MemoryEntry(
            type=MemoryType.SUMMARY,
            messages=[{"role": "system", "content": "Session memory summary:\nsummary"}],
            content="summary",
            metadata={"source_entry_count": len(entries)},
        )


class FakeSummaryClient:
    def __init__(self, responses: list[str]) -> None:
        self.responses = responses
        self.requests: list[list[dict[str, str]]] = []

    def chat(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        tools: list[dict[str, str]],
        stream: bool = False,
    ) -> AssistantMessage:
        self.requests.append(messages)
        return AssistantMessage(
            content=self.responses.pop(0),
            tool_calls=[],
            raw={},
            token_usage=TokenUsage(),
        )


class SessionMemoryTests(unittest.TestCase):
    def test_fifo_moves_old_entries_for_later_compression(self) -> None:
        memory = SessionMemory({"role": "system", "content": "system"}, max_active_entries=2)

        memory.add_conversation({"role": "user", "content": "first"}, content="first")
        memory.add_conversation({"role": "assistant", "content": "second"}, content="second")
        memory.add_conversation({"role": "user", "content": "third"}, content="third")

        self.assertEqual([entry.content for entry in memory.active_entries], ["second", "third"])
        self.assertEqual([entry.content for entry in memory.old_entries], ["first"])

    def test_compress_old_entries_retains_summary(self) -> None:
        memory = SessionMemory({"role": "system", "content": "system"}, max_active_entries=1)
        memory.add_conversation({"role": "user", "content": "first"}, content="first")
        memory.add_conversation({"role": "assistant", "content": "second"}, content="second")

        summary = memory.compress_old_entries(StaticCompressor())

        self.assertIsNotNone(summary)
        assert summary is not None
        self.assertEqual(summary.type, MemoryType.SUMMARY)
        self.assertEqual(memory.old_entries, [])
        self.assertIn("summary", memory.messages()[1]["content"])

    def test_fifo_keeps_native_tool_call_pairs_together(self) -> None:
        memory = SessionMemory({"role": "system", "content": "system"}, max_active_entries=1)
        memory.add_conversation(
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "list_files", "arguments": "{}"},
                    }
                ],
            }
        )
        memory.add_tool_result(
            {"role": "tool", "tool_call_id": "call_1", "content": "README.md"},
            tool_name="list_files",
            ok=True,
            content="README.md",
        )

        self.assertEqual(memory.active_entries, [])
        self.assertEqual([entry.type for entry in memory.old_entries], [MemoryType.CONVERSATION, MemoryType.TOOL_RESULT])

    def test_token_usage_ratio_uses_context_window(self) -> None:
        self.assertEqual(token_usage_ratio(80, 100), 0.8)
        self.assertIsNone(token_usage_ratio(None, 100))

    def test_move_entries_before_recent_rounds_keeps_recent_three_rounds(self) -> None:
        memory = SessionMemory({"role": "system", "content": "system"}, max_active_entries=100)
        for index in range(1, 6):
            memory.add_conversation({"role": "user", "content": f"user {index}"}, content=f"user {index}")
            memory.add_conversation(
                {"role": "assistant", "content": f"assistant {index}"},
                content=f"assistant {index}",
            )

        moved = memory.move_entries_before_recent_rounds_to_old(3)

        self.assertEqual(moved, 4)
        self.assertEqual([entry.content for entry in memory.old_entries], ["user 1", "assistant 1", "user 2", "assistant 2"])
        self.assertEqual(memory.active_entries[0].content, "user 3")


class MapReduceContextCompressorTests(unittest.TestCase):
    def test_single_chunk_uses_map_summary_without_reduce_and_saves_facts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            memory = LongTermMemory(Path(tmp) / ".mikucli" / "long_term_memory.json")
            client = FakeSummaryClient(
                [
                    "mapped summary",
                    '["User prefers concise answers.", "Project uses PowerShell."]',
                ]
            )
            compressor = MapReduceContextCompressor(
                client=client,
                model="test-model",
                long_term_memory=memory,
                chunk_chars=1000,
            )

            summary = compressor.compress(
                [
                    MemoryEntry(
                        type=MemoryType.CONVERSATION,
                        messages=[{"role": "user", "content": "hello"}],
                        content="hello",
                    )
                ]
            )

            self.assertIsNotNone(summary)
            assert summary is not None
            self.assertIn("mapped summary", summary.content)
            self.assertEqual(len(client.requests), 2)
            self.assertEqual(summary.metadata["map_chunk_count"], 1)
            self.assertEqual(summary.metadata["saved_fact_count"], 2)
            self.assertEqual([record.content for record in memory.records], ["User prefers concise answers.", "Project uses PowerShell."])

    def test_multiple_chunks_are_reduced_after_mapping(self) -> None:
        client = FakeSummaryClient(["map one", "map two", "reduced", "[]"])
        compressor = MapReduceContextCompressor(
            client=client,
            model="test-model",
            chunk_chars=15,
        )

        summary = compressor.compress(
            [
                MemoryEntry(type=MemoryType.CONVERSATION, messages=[], content="first chunk text"),
                MemoryEntry(type=MemoryType.CONVERSATION, messages=[], content="second chunk text"),
            ]
        )

        self.assertIsNotNone(summary)
        assert summary is not None
        self.assertIn("reduced", summary.content)
        self.assertEqual(len(client.requests), 4)
        self.assertEqual(summary.metadata["map_chunk_count"], 2)

    def test_fact_extraction_accepts_json_array_inside_text(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            memory = LongTermMemory(Path(tmp) / ".mikucli" / "long_term_memory.json")
            client = FakeSummaryClient(
                [
                    "mapped summary",
                    'Facts:\n["User prefers concise answers."]',
                ]
            )
            compressor = MapReduceContextCompressor(
                client=client,
                model="test-model",
                long_term_memory=memory,
            )

            compressor.compress(
                [MemoryEntry(type=MemoryType.CONVERSATION, messages=[], content="old")]
            )

            self.assertEqual([record.content for record in memory.records], ["User prefers concise answers."])


class LongTermMemoryTests(unittest.TestCase):
    def test_save_writes_long_term_memory_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / ".mikucli" / "long_term_memory.json"
            memory = LongTermMemory(path)

            result = memory.save("User prefers concise answers.")

            self.assertTrue(result.saved)
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(payload["memories"][0]["content"], "User prefers concise answers.")
            self.assertEqual(payload["memories"][0]["created_at"], result.record.created_at)

    def test_duplicate_save_reuses_original_timestamp(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / ".mikucli" / "long_term_memory.json"
            path.parent.mkdir(parents=True)
            path.write_text(
                json.dumps(
                    {
                        "memories": [
                            {
                                "content": "User prefers concise answers.",
                                "created_at": "2026-01-01T00:00:00+00:00",
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            memory = LongTermMemory(path)

            result = memory.save(" user   prefers CONCISE answers. ")

            self.assertFalse(result.saved)
            self.assertEqual(result.record.created_at, "2026-01-01T00:00:00+00:00")
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(len(payload["memories"]), 1)

    def test_session_memory_includes_loaded_long_term_memory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = default_long_term_memory_path(Path(tmp))
            memory = LongTermMemory(path)
            memory.save("User works in PowerShell.")
            session_memory = SessionMemory(
                {"role": "system", "content": "system"},
                long_term_memory=memory,
            )

            messages = session_memory.messages()

            self.assertIn("Retrieved memory", messages[1]["content"])
            self.assertIn("User works in PowerShell.", messages[1]["content"])


class MemoryRetrieverTests(unittest.TestCase):
    def test_keyword_matchup_scores_query_overlap(self) -> None:
        retriever = MemoryRetriever()

        score = retriever.keyword_matchup({"python", "tests"}, "Python unit tests passed")

        self.assertEqual(score, 1.0)

    def test_time_decay_decreases_linearly_to_half_over_24_hours(self) -> None:
        now = datetime(2026, 1, 2, tzinfo=timezone.utc)
        retriever = MemoryRetriever(now=lambda: now)

        self.assertEqual(retriever.time_decay("2026-01-02T00:00:00+00:00"), 1.0)
        self.assertEqual(retriever.time_decay("2026-01-01T12:00:00+00:00"), 0.75)
        self.assertEqual(retriever.time_decay("2026-01-01T00:00:00+00:00"), 0.5)
        self.assertEqual(retriever.time_decay("2025-12-31T00:00:00+00:00"), 0.5)

    def test_long_term_memory_has_source_weight_multiplier(self) -> None:
        now = datetime(2026, 1, 2, tzinfo=timezone.utc)
        retriever = MemoryRetriever(now=lambda: now)

        memories = retriever.retrieve(
            query="python",
            session_entries=[
                MemoryEntry(
                    type=MemoryType.SUMMARY,
                    messages=[],
                    content="python",
                    created_at="2026-01-02T00:00:00+00:00",
                )
            ],
            long_term_records=[
                LongTermMemoryRecord("python", "2026-01-02T00:00:00+00:00"),
            ],
        )

        self.assertEqual(memories[0].source, "long_term")
        self.assertEqual(memories[0].score, 1.2)
        self.assertEqual(memories[1].score, 1.0)

    def test_session_memory_sends_retrieved_memories_for_query(self) -> None:
        now = datetime(2026, 1, 2, tzinfo=timezone.utc)
        retriever = MemoryRetriever(now=lambda: now)
        with tempfile.TemporaryDirectory() as tmp:
            memory = LongTermMemory(Path(tmp) / ".mikucli" / "long_term_memory.json")
            memory.save("User likes Python examples.")
            session_memory = SessionMemory(
                {"role": "system", "content": "system"},
                long_term_memory=memory,
                retriever=retriever,
            )
            session_memory.summary_entries.append(
                MemoryEntry(
                    type=MemoryType.SUMMARY,
                    messages=[{"role": "system", "content": "Session memory summary:\nPowerShell notes"}],
                    content="PowerShell notes",
                    created_at="2026-01-02T00:00:00+00:00",
                )
            )

            messages = session_memory.messages(query="python")

            self.assertIn("Retrieved memory", messages[1]["content"])
            self.assertIn("User likes Python examples.", messages[1]["content"])
            self.assertNotIn("PowerShell notes", messages[1]["content"])


if __name__ == "__main__":
    unittest.main()

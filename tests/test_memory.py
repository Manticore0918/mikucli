from __future__ import annotations

import json
import math
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
from mikucli.memory.compression import CompressionProposal, CompressionValidationError
from mikucli.memory.ledger import entry_checksum


class StaticCompressor:
    def prepare(self, entries: list[MemoryEntry], protected=(), *, scope_id=""):
        if not entries:
            return None
        summary = MemoryEntry(
            type=MemoryType.SUMMARY,
            messages=[{"role": "system", "content": "Session memory summary:\nsummary"}],
            content="summary",
            metadata={"source_entry_count": len(entries)},
        )
        return CompressionProposal(summary, {e.entry_id: entry_checksum(e) for e in entries}, protected)


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


class FakeEmbeddingClient:
    model = "test-embedding"

    def __init__(self, vectors: dict[str, list[float]]) -> None:
        self.vectors = vectors
        self.requests: list[list[str]] = []

    def embed(self, inputs: list[str]) -> list[list[float]]:
        self.requests.append(list(inputs))
        return [self.vectors[value] for value in inputs]


class FailingEmbeddingClient:
    model = "test-embedding"

    def embed(self, inputs: list[str]) -> list[list[float]]:
        raise RuntimeError("embedding service offline")


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
            text = "User prefers concise answers. Project uses PowerShell."
            entry = MemoryEntry(type=MemoryType.CONVERSATION, messages=[{"role": "user", "content": text}])
            client = FakeSummaryClient([json.dumps({"summary": "mapped summary",
                "covered_parts": [entry.entry_id + ":0"], "records": [], "durable_facts": [
                    {"source_id": entry.entry_id, "quote": quote, "scope": "workspace"}
                    for quote in ("User prefers concise answers.", "Project uses PowerShell.")
                ]})])
            compressor = MapReduceContextCompressor(
                client=client,
                model="test-model",
                long_term_memory=memory,
                chunk_chars=1000,
            )

            summary = compressor.compress([entry])

            self.assertIsNotNone(summary)
            assert summary is not None
            self.assertIn("mapped summary", summary.content)
            self.assertEqual(len(client.requests), 1)
            self.assertEqual(summary.metadata["map_chunk_count"], 1)
            self.assertEqual(summary.metadata["saved_fact_count"], 2)
            self.assertEqual([record.content for record in memory.records], ["User prefers concise answers.", "Project uses PowerShell."])

    def test_multiple_chunks_are_reduced_after_mapping(self) -> None:
        entries = [MemoryEntry(type=MemoryType.CONVERSATION, messages=[], content=text)
                   for text in ("first chunk text", "second chunk text")]
        client = FakeSummaryClient([*(json.dumps({"summary": f"map {i}",
            "covered_parts": [entry.entry_id + ":0"], "records": [], "durable_facts": []})
            for i, entry in enumerate(entries)), "reduced"])
        compressor = MapReduceContextCompressor(
            client=client,
            model="test-model",
            chunk_chars=32,
        )

        summary = compressor.compress(entries)

        self.assertIsNotNone(summary)
        assert summary is not None
        self.assertIn("reduced", summary.content)
        self.assertEqual(len(client.requests), 3)
        self.assertEqual(summary.metadata["map_chunk_count"], 2)

    def test_malformed_fact_output_is_rejected_without_saving_facts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            memory = LongTermMemory(Path(tmp) / ".mikucli" / "long_term_memory.json")
            client = FakeSummaryClient(
                [
                    'Facts:\n["User prefers concise answers."]',
                    'Facts:\n["User prefers concise answers."]',
                ]
            )
            compressor = MapReduceContextCompressor(
                client=client,
                model="test-model",
                long_term_memory=memory,
            )

            with self.assertRaises(CompressionValidationError):
                compressor.compress([MemoryEntry(type=MemoryType.CONVERSATION, messages=[], content="old")])

            self.assertEqual(memory.records, [])


class LongTermMemoryTests(unittest.TestCase):
    def test_save_commits_long_term_memory_database(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / ".mikucli" / "long_term_memory.json"
            memory = LongTermMemory(path)

            result = memory.save("User prefers concise answers.")

            self.assertTrue(result.saved)
            self.assertFalse(path.exists())
            reopened = LongTermMemory(memory.path)
            self.assertEqual(reopened.records, [result.record])

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
            self.assertEqual(LongTermMemory(memory.path).records, [result.record])

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
        retriever = MemoryRetriever(
            embedding_client=FakeEmbeddingClient({}),
            embedding_model="test-embedding",
        )

        score = retriever.keyword_matchup({"python", "tests"}, "Python unit tests passed")

        self.assertEqual(score, 1.0)

    def test_time_decay_decreases_linearly_to_half_over_24_hours(self) -> None:
        now = datetime(2026, 1, 2, tzinfo=timezone.utc)
        retriever = MemoryRetriever(
            embedding_client=FakeEmbeddingClient({}),
            embedding_model="test-embedding",
            now=lambda: now,
        )

        self.assertEqual(retriever.time_decay("2026-01-02T00:00:00+00:00"), 1.0)
        self.assertEqual(retriever.time_decay("2026-01-01T12:00:00+00:00"), 0.75)
        self.assertEqual(retriever.time_decay("2026-01-01T00:00:00+00:00"), 0.5)
        self.assertEqual(retriever.time_decay("2025-12-31T00:00:00+00:00"), 0.5)

    def test_long_term_memory_has_source_weight_multiplier(self) -> None:
        now = datetime(2026, 1, 2, tzinfo=timezone.utc)
        retriever = MemoryRetriever(
            embedding_client=FakeEmbeddingClient({"python": [1.0, 0.0]}),
            embedding_model="test-embedding",
            now=lambda: now,
        )

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

    def test_hybrid_score_blends_keyword_and_semantic_relevance_equally(self) -> None:
        now = datetime(2026, 1, 2, tzinfo=timezone.utc)
        retriever = MemoryRetriever(
            embedding_client=FakeEmbeddingClient({}),
            embedding_model="test-embedding",
            now=lambda: now,
        )

        keyword_only = retriever.score(
            query_terms={"python"},
            content="python",
            semantic_similarity=0.0,
            created_at="2026-01-02T00:00:00+00:00",
            source="session",
        )
        semantic_only = retriever.score(
            query_terms={"python"},
            content="snake language",
            semantic_similarity=1.0,
            created_at="2026-01-02T00:00:00+00:00",
            source="session",
        )

        self.assertEqual(keyword_only, 0.5)
        self.assertEqual(semantic_only, 0.5)

    def test_semantic_only_matches_require_similarity_of_at_least_half(self) -> None:
        now = datetime(2026, 1, 2, tzinfo=timezone.utc)
        retriever = MemoryRetriever(
            embedding_client=FakeEmbeddingClient(
                {
                    "deployment": [1.0, 0.0],
                    "release pipeline": [0.5, math.sqrt(0.75)],
                    "lunch order": [0.49, math.sqrt(1.0 - (0.49**2))],
                }
            ),
            embedding_model="test-embedding",
            now=lambda: now,
        )

        memories = retriever.retrieve(
            query="deployment",
            session_entries=[
                MemoryEntry(
                    type=MemoryType.SUMMARY,
                    messages=[],
                    content="release pipeline",
                    created_at="2026-01-02T00:00:00+00:00",
                ),
                MemoryEntry(
                    type=MemoryType.SUMMARY,
                    messages=[],
                    content="lunch order",
                    created_at="2026-01-02T00:00:00+00:00",
                ),
            ],
            long_term_records=[],
        )

        self.assertEqual([memory.content for memory in memories], ["release pipeline"])

    def test_lazy_embeddings_are_cached_for_repeated_retrieval(self) -> None:
        client = FakeEmbeddingClient(
            {
                "python": [1.0, 0.0],
                "Python examples": [1.0, 0.0],
            }
        )
        retriever = MemoryRetriever(
            embedding_client=client,
            embedding_model="test-embedding",
        )
        records = [LongTermMemoryRecord("Python examples", "2026-01-02T00:00:00+00:00")]

        retriever.retrieve(query="python", session_entries=[], long_term_records=records)
        retriever.retrieve(query="python", session_entries=[], long_term_records=records)

        self.assertEqual(client.requests, [["python", "Python examples"]])

    def test_nomic_memory_embeddings_use_query_and_document_prefixes(self) -> None:
        client = FakeEmbeddingClient(
            {
                "search_query: python": [1.0, 0.0],
                "search_document: Python examples": [1.0, 0.0],
            }
        )
        retriever = MemoryRetriever(
            embedding_client=client,
            embedding_model="nomic-embed-text",
        )

        retriever.retrieve(
            query="python",
            session_entries=[],
            long_term_records=[
                LongTermMemoryRecord("Python examples", "2026-01-02T00:00:00+00:00")
            ],
        )

        self.assertEqual(
            client.requests,
            [["search_query: python", "search_document: Python examples"]],
        )

    def test_semantic_embedding_failure_is_not_downgraded_to_keyword_only(self) -> None:
        retriever = MemoryRetriever(
            embedding_client=FailingEmbeddingClient(),
            embedding_model="test-embedding",
        )

        with self.assertRaisesRegex(RuntimeError, "embedding service offline"):
            retriever.retrieve(
                query="python",
                session_entries=[],
                long_term_records=[
                    LongTermMemoryRecord("python", "2026-01-02T00:00:00+00:00")
                ],
            )

    def test_session_memory_sends_retrieved_memories_for_query(self) -> None:
        now = datetime(2026, 1, 2, tzinfo=timezone.utc)
        retriever = MemoryRetriever(
            embedding_client=FakeEmbeddingClient(
                {
                    "python": [1.0, 0.0],
                    "User likes Python examples.": [1.0, 0.0],
                    "PowerShell notes": [0.0, 1.0],
                }
            ),
            embedding_model="test-embedding",
            now=lambda: now,
        )
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

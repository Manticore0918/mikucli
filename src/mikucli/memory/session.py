from __future__ import annotations

from typing import Any

from .long_term import LongTermMemory
from .models import ContextCompressor, MemoryEntry, MemoryType, RetrievedMemory
from .retrieval import MemoryRetriever
from .utilities import entry_content, entry_has_tool_calls, recent_round_boundary


class SessionMemory:
    """Maintain active, old, summarized, and retrieved session context."""

    def __init__(
        self,
        system_message: dict[str, Any],
        max_active_entries: int = 40,
        long_term_memory: LongTermMemory | None = None,
        retriever: MemoryRetriever | None = None,
        retrieved_memory_limit: int = 8,
    ) -> None:
        if max_active_entries <= 0:
            raise ValueError("max_active_entries must be positive")
        self.system_message = system_message
        self.max_active_entries = max_active_entries
        self.long_term_memory = long_term_memory
        self.retriever = retriever or MemoryRetriever()
        self.retrieved_memory_limit = retrieved_memory_limit
        self.active_entries: list[MemoryEntry] = []
        self.old_entries: list[MemoryEntry] = []
        self.summary_entries: list[MemoryEntry] = []

    def add_conversation(self, message: dict[str, Any], content: str = "") -> None:
        self._add(MemoryEntry(type=MemoryType.CONVERSATION, messages=[message], content=content))

    def add_fact(self, content: str) -> None:
        self._add(
            MemoryEntry(
                type=MemoryType.FACT,
                messages=[{"role": "system", "content": f"Session fact: {content}"}],
                content=content,
            )
        )

    def add_tool_result(
        self,
        message: dict[str, Any],
        *,
        tool_name: str,
        ok: bool,
        content: str,
    ) -> None:
        self._add(
            MemoryEntry(
                type=MemoryType.TOOL_RESULT,
                messages=[message],
                content=content,
                metadata={"tool": tool_name, "ok": ok},
            )
        )

    def messages(self, query: str | None = None, *, system_overlay: str = "") -> list[dict[str, Any]]:
        system_message = dict(self.system_message)
        if system_overlay:
            base_content = str(system_message.get("content", "")).rstrip()
            system_message["content"] = f"{base_content}\n\n{system_overlay}"
        return [
            system_message,
            *self._retrieved_memory_messages(query),
            *(message for entry in self.active_entries for message in entry.messages),
        ]

    def compress_old_entries(self, compressor: ContextCompressor) -> MemoryEntry | None:
        summary = compressor.compress(self.old_entries)
        if summary is None:
            return None
        self.summary_entries.append(summary)
        self.old_entries.clear()
        return summary

    def prepare_entries_for_compression(self, retain_rounds: int) -> int:
        if retain_rounds < 0:
            raise ValueError("retain_rounds must be non-negative")
        combined = [*self.old_entries, *self.active_entries]
        boundary = recent_round_boundary(combined, retain_rounds)
        self.old_entries = combined[:boundary]
        self.active_entries = combined[boundary:]
        return len(self.old_entries)

    def move_entries_before_recent_rounds_to_old(self, retain_rounds: int) -> int:
        if retain_rounds < 0:
            raise ValueError("retain_rounds must be non-negative")
        boundary = recent_round_boundary(self.active_entries, retain_rounds)
        moving = self.active_entries[:boundary]
        if not moving:
            return 0
        self.old_entries.extend(moving)
        del self.active_entries[:boundary]
        return len(moving)

    def _add(self, entry: MemoryEntry) -> None:
        self.active_entries.append(entry)
        self._enforce_fifo_limit()

    def _enforce_fifo_limit(self) -> None:
        while len(self.active_entries) > self.max_active_entries:
            self._move_oldest_batch()
        while self.active_entries and self.active_entries[0].type == MemoryType.TOOL_RESULT:
            self.old_entries.append(self.active_entries.pop(0))

    def _move_oldest_batch(self) -> None:
        first = self.active_entries.pop(0)
        self.old_entries.append(first)
        if entry_has_tool_calls(first):
            while self.active_entries and self.active_entries[0].type == MemoryType.TOOL_RESULT:
                self.old_entries.append(self.active_entries.pop(0))

    def _retrieved_memory_messages(self, query: str | None) -> list[dict[str, Any]]:
        session_entries = [*self.summary_entries, *self.old_entries]
        long_term_records = self.long_term_memory.records if self.long_term_memory is not None else []
        if not session_entries and not long_term_records:
            return []
        if query:
            memories = self.retriever.retrieve(
                query=query,
                session_entries=session_entries,
                long_term_records=long_term_records,
                limit=self.retrieved_memory_limit,
            )
        else:
            memories = [
                RetrievedMemory(
                    source="session",
                    content=entry_content(entry),
                    created_at=entry.created_at,
                    score=0.0,
                )
                for entry in session_entries
                if entry_content(entry)
            ]
            memories.extend(
                RetrievedMemory(
                    source="long_term",
                    content=record.content,
                    created_at=record.created_at,
                    score=0.0,
                )
                for record in long_term_records
            )
        if not memories:
            return []
        lines = [
            f"- {memory.source} {memory.created_at} score={memory.score:.3f}: {memory.content}"
            for memory in memories
        ]
        return [{"role": "system", "content": "Retrieved memory:\n" + "\n".join(lines)}]

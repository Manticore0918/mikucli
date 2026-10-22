from __future__ import annotations

from copy import deepcopy
from typing import Any
from uuid import uuid4

from .archive import SessionArchive
from .compression import CompressionValidationError
from .ledger import ProtectedRecord, TaskLedger, evidence_text, source_role
from .long_term import LongTermMemory
from .models import ContextCompressor, MemoryEntry, MemoryType, RetrievedMemory
from .retrieval import MemoryRetrievalError, MemoryRetriever
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
        archive: SessionArchive | None = None,
    ) -> None:
        if max_active_entries <= 0:
            raise ValueError("max_active_entries must be positive")
        self.system_message = system_message
        self.max_active_entries = max_active_entries
        self.long_term_memory = long_term_memory
        self.retriever = retriever
        self.retrieved_memory_limit = retrieved_memory_limit
        self.active_entries: list[MemoryEntry] = []
        self.old_entries: list[MemoryEntry] = []
        self.summary_entries: list[MemoryEntry] = []
        self.scope_id = uuid4().hex
        self.ledger = TaskLedger(self.scope_id)
        self.archive = archive or SessionArchive()
        self._entry_order: dict[str, int] = {}

    def add_conversation(self, message: dict[str, Any], content: str = "") -> None:
        self._add(
            MemoryEntry(
                type=MemoryType.CONVERSATION, messages=[message], content=content
            )
        )

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

    def messages(
        self,
        query: str | None = None,
        *,
        system_overlay: str = "",
        context_window_tokens: int | None = None,
        tools: list[dict] | None = None,
        include_retrieved: bool = True,
    ) -> list[dict[str, Any]]:
        system_message = dict(self.system_message)
        if system_overlay:
            base_content = str(system_message.get("content", "")).rstrip()
            system_message["content"] = f"{base_content}\n\n{system_overlay}"
        protected = self.ledger.message()
        required = [
            system_message,
            *([protected] if protected else []),
            *(message for entry in self.active_entries for message in entry.messages),
        ]
        if context_window_tokens is not None:
            from .prompt_budget import check_prompt

            check_prompt(required, tools or [], context_window_tokens)
        retrieved = (
            self._retrieved_memory_messages(
                query,
                required=required,
                tools=tools or [],
                context_window_tokens=context_window_tokens,
            )
            if include_retrieved
            else []
        )
        return [system_message, *retrieved, *required[1:]]

    def compress_old_entries(self, compressor: ContextCompressor) -> MemoryEntry | None:
        entries = deepcopy(self.old_entries)
        if not entries:
            return None
        for entry in [*entries, *self.active_entries]:
            self._capture(entry)
        protected = self.ledger.snapshot()
        # Journal exact inputs before model calls, so a crash mid-compression is recoverable.
        checkpoint = MemoryEntry(
            MemoryType.SUMMARY,
            [],
            content="Compression input checkpoint.",
            metadata={"kind": "compression_input"},
        )
        from mikucli.run_control import check_active, current_budget

        check_active()
        self.archive.commit_batch(
            self.scope_id, [*entries, *self.active_entries], [], checkpoint, protected
        )
        proposal = compressor.prepare(entries, protected, scope_id=self.scope_id)
        if proposal is None:
            raise CompressionValidationError(
                "Nonempty source batch produced no compression proposal."
            )
        proposal.validate(self.old_entries, self.ledger.snapshot())
        new_ledger = TaskLedger(self.scope_id)
        for record in proposal.protected:
            new_ledger.add(record)
            source = self.get_source(record.source_entry_id)
            expected = ProtectedRecord.from_source(
                record.kind, record.quote, source, self.scope_id, origin=record.origin
            )
            if (
                record.record_id != expected.record_id
                or record.source_role != source_role(source)
                or record.quote not in evidence_text(source)
                or (record not in protected and record.origin != "extracted")
            ):
                raise CompressionValidationError(
                    "Protected state references invalid evidence."
                )
        summary = proposal.summary
        check_active()
        if current_budget() is not None:
            current_budget().check()
        batch_id = self.archive.commit_batch(
            self.scope_id,
            [*entries, *self.active_entries],
            [entry.entry_id for entry in entries],
            summary,
            new_ledger.snapshot(),
        )
        memory = self.long_term_memory or getattr(compressor, "long_term_memory", None)
        if memory is not None:
            summary.metadata["saved_fact_count"] = sum(
                r.saved for r in memory.save_many(list(proposal.facts))
            )
        summary.metadata["archive_batch_id"] = batch_id
        summary.metadata["scope_id"] = self.scope_id
        check_active()
        if current_budget() is not None:
            current_budget().check()
        proposal.validate(self.old_entries, self.ledger.snapshot())
        self.ledger = new_ledger
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
        self._capture(entry)
        self.active_entries.append(entry)
        self._enforce_fifo_limit()

    def _capture(self, entry: MemoryEntry) -> None:
        if entry.entry_id not in self._entry_order:
            self._entry_order[entry.entry_id] = len(self._entry_order)
            self.ledger.capture(entry)

    def get_source(self, entry_id: str) -> MemoryEntry:
        for entry in [*self.old_entries, *self.active_entries]:
            if entry.entry_id == entry_id:
                return entry
        return self.archive.get_source(self.scope_id, entry_id)

    def update_task_state(
        self, record_id: str, status: str, source_id: str, quote: str
    ) -> None:
        evidence = self.get_source(source_id)
        record = self.ledger.records[record_id]
        if self._entry_order.get(source_id, -1) < self._entry_order.get(
            record.source_entry_id, -1
        ):
            raise ValueError("Status evidence must not predate the protected record.")
        candidate = TaskLedger(self.scope_id)
        for existing in self.ledger.snapshot():
            candidate.add(existing)
        candidate.set_status(record_id, status, evidence, quote)
        checkpoint = MemoryEntry(
            MemoryType.SUMMARY,
            [],
            content="Source-backed task-state lifecycle update.",
            metadata={"kind": "ledger_update"},
        )
        self.archive.commit_batch(
            self.scope_id,
            [*self.old_entries, *self.active_entries],
            [],
            checkpoint,
            candidate.snapshot(),
        )
        self.ledger = candidate

    def clear(self) -> None:
        self.active_entries.clear()
        self.old_entries.clear()
        self.summary_entries.clear()
        self.scope_id = uuid4().hex
        self.ledger = TaskLedger(self.scope_id)
        self._entry_order.clear()

    def _enforce_fifo_limit(self) -> None:
        while len(self.active_entries) > self.max_active_entries:
            self._move_oldest_batch()
        while (
            self.active_entries
            and self.active_entries[0].type == MemoryType.TOOL_RESULT
        ):
            self.old_entries.append(self.active_entries.pop(0))

    def _move_oldest_batch(self) -> None:
        first = self.active_entries.pop(0)
        self.old_entries.append(first)
        if entry_has_tool_calls(first):
            while (
                self.active_entries
                and self.active_entries[0].type == MemoryType.TOOL_RESULT
            ):
                self.old_entries.append(self.active_entries.pop(0))

    def _retrieved_memory_messages(
        self,
        query: str | None,
        *,
        required: list[dict] | None = None,
        tools: list[dict] | None = None,
        context_window_tokens: int | None = None,
    ) -> list[dict[str, Any]]:
        session_entries = [*self.summary_entries, *self.old_entries]
        long_term_records = (
            self.long_term_memory.records if self.long_term_memory is not None else []
        )
        if not session_entries and not long_term_records:
            return []
        if query:
            if self.retriever is None:
                raise MemoryRetrievalError(
                    "Hybrid Memory Retrieval is not configured for this agent session."
                )
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
        lines: list[str] = []
        for memory in memories:
            block = f"- {memory.source} {memory.created_at} score={memory.score:.3f}: {memory.content}"
            if context_window_tokens is not None:
                from .prompt_budget import fits_prompt

                candidate = {
                    "role": "system",
                    "content": "Retrieved memory:\n" + "\n".join([*lines, block]),
                }
                if not fits_prompt(
                    [*(required or []), candidate], tools or [], context_window_tokens
                ):
                    continue
            lines.append(block)
        if not lines:
            return []
        return [{"role": "system", "content": "Retrieved memory:\n" + "\n".join(lines)}]

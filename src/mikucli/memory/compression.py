from __future__ import annotations

from .long_term import LongTermMemory
from .models import MemoryEntry, MemoryType, SummaryClient
from .utilities import chunk_memory_entries, parse_json_string_list


class MapReduceContextCompressor:
    """Compress older session entries and promote durable facts."""

    def __init__(
        self,
        *,
        client: SummaryClient,
        model: str,
        long_term_memory: LongTermMemory | None = None,
        chunk_chars: int = 6000,
    ) -> None:
        if chunk_chars <= 0:
            raise ValueError("chunk_chars must be positive")
        self.client = client
        self.model = model
        self.long_term_memory = long_term_memory
        self.chunk_chars = chunk_chars

    def compress(self, entries: list[MemoryEntry]) -> MemoryEntry | None:
        if not entries:
            return None
        chunks = chunk_memory_entries(entries, self.chunk_chars)
        if not chunks:
            return None
        mapped = [self._summarize_chunk(chunk) for chunk in chunks]
        mapped = [summary.strip() for summary in mapped if summary.strip()]
        if not mapped:
            return None
        content = mapped[0] if len(mapped) == 1 else self._reduce_summaries(mapped)
        facts = self._extract_facts(content)
        saved_fact_count = self._save_facts(facts)
        summary = f"Compressed prior session memory:\n{content}"
        return MemoryEntry(
            type=MemoryType.SUMMARY,
            messages=[{"role": "system", "content": f"Session memory summary:\n{content}"}],
            content=summary,
            metadata={
                "source_entry_count": len(entries),
                "map_chunk_count": len(chunks),
                "saved_fact_count": saved_fact_count,
            },
        )

    def _summarize_chunk(self, chunk: str) -> str:
        return self._ask_llm(
            "Summarize this older session-memory chunk for future context. "
            "Preserve user goals, project state, tool outcomes, unresolved tasks, and decisions. "
            "Do not include filler.",
            chunk,
        )

    def _reduce_summaries(self, summaries: list[str]) -> str:
        return self._ask_llm(
            "Merge these mapped session-memory summaries into one concise context summary. "
            "Remove duplication while preserving important project state, decisions, and open work.",
            "\n\n".join(f"Summary {index + 1}:\n{summary}" for index, summary in enumerate(summaries)),
        )

    def _extract_facts(self, summary: str) -> list[str]:
        raw = self._ask_llm(
            "Extract durable facts from this compressed context for long-term memory. "
            "Focus on user preferences, project settings, important decisions, and stable project facts. "
            "Return only a JSON array of strings. Return [] when there are no durable facts.",
            summary,
        )
        return parse_json_string_list(raw)

    def _save_facts(self, facts: list[str]) -> int:
        if self.long_term_memory is None:
            return 0
        saved = 0
        for fact in facts:
            result = self.long_term_memory.save(fact)
            if result.saved:
                saved += 1
        return saved

    def _ask_llm(self, instruction: str, content: str) -> str:
        response = self.client.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": "You compress agent session memory without inventing facts."},
                {"role": "user", "content": f"{instruction}\n\n{content}"},
            ],
            tools=[],
        )
        return str(getattr(response, "content", ""))

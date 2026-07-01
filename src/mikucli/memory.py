from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
import re
from typing import Any, Callable, Literal, Protocol


class MemoryType(str, Enum):
    CONVERSATION = "CONVERSATION"
    FACT = "FACT"
    SUMMARY = "SUMMARY"
    TOOL_RESULT = "TOOL_RESULT"


@dataclass
class MemoryEntry:
    type: MemoryType
    messages: list[dict[str, Any]]
    content: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: _now())


@dataclass(frozen=True)
class LongTermMemoryRecord:
    content: str
    created_at: str


@dataclass(frozen=True)
class LongTermMemorySaveResult:
    record: LongTermMemoryRecord
    saved: bool


@dataclass(frozen=True)
class RetrievedMemory:
    source: Literal["session", "long_term"]
    content: str
    created_at: str
    score: float


class ContextCompressor(Protocol):
    def compress(self, entries: list[MemoryEntry]) -> MemoryEntry | None: ...


class SummaryClient(Protocol):
    def chat(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        stream: bool = False,
    ) -> Any: ...


class MapReduceContextCompressor:
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

        chunks = _chunk_memory_entries(entries, self.chunk_chars)
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
        return _parse_json_string_list(raw)

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


class MemoryRetriever:
    def __init__(
        self,
        *,
        now: Callable[[], datetime] | None = None,
        long_term_source_weight: float = 1.2,
        session_source_weight: float = 1.0,
        time_decay_window: timedelta = timedelta(hours=24),
        min_time_decay: float = 0.5,
    ) -> None:
        self.now = now or (lambda: datetime.now(timezone.utc))
        self.long_term_source_weight = long_term_source_weight
        self.session_source_weight = session_source_weight
        self.time_decay_window = time_decay_window
        self.min_time_decay = min_time_decay

    def retrieve(
        self,
        *,
        query: str,
        session_entries: list[MemoryEntry],
        long_term_records: list[LongTermMemoryRecord],
        limit: int = 8,
    ) -> list[RetrievedMemory]:
        if limit <= 0:
            return []

        query_terms = _keywords(query)
        candidates: list[RetrievedMemory] = []
        for entry in session_entries:
            content = _entry_content(entry)
            if not content:
                continue
            score = self.score(
                query_terms=query_terms,
                content=content,
                created_at=entry.created_at,
                source="session",
            )
            if score > 0:
                candidates.append(
                    RetrievedMemory(
                        source="session",
                        content=content,
                        created_at=entry.created_at,
                        score=score,
                    )
                )

        for record in long_term_records:
            score = self.score(
                query_terms=query_terms,
                content=record.content,
                created_at=record.created_at,
                source="long_term",
            )
            if score > 0:
                candidates.append(
                    RetrievedMemory(
                        source="long_term",
                        content=record.content,
                        created_at=record.created_at,
                        score=score,
                    )
                )

        return sorted(
            candidates,
            key=lambda candidate: (
                candidate.score,
                _parse_timestamp(candidate.created_at),
                1 if candidate.source == "long_term" else 0,
            ),
            reverse=True,
        )[:limit]

    def score(
        self,
        *,
        query_terms: set[str],
        content: str,
        created_at: str,
        source: Literal["session", "long_term"],
    ) -> float:
        keyword_score = self.keyword_matchup(query_terms, content)
        if keyword_score <= 0:
            return 0.0
        return keyword_score * self.time_decay(created_at) * self.source_weight(source)

    def keyword_matchup(self, query_terms: set[str], content: str) -> float:
        if not query_terms:
            return 0.0
        content_terms = _keywords(content)
        if not content_terms:
            return 0.0
        return len(query_terms & content_terms) / len(query_terms)

    def time_decay(self, created_at: str) -> float:
        created = _parse_timestamp(created_at)
        if created is None:
            return self.min_time_decay
        age = self.now() - created
        if age.total_seconds() <= 0:
            return 1.0
        ratio = min(age / self.time_decay_window, 1.0)
        return 1.0 - ((1.0 - self.min_time_decay) * ratio)

    def source_weight(self, source: Literal["session", "long_term"]) -> float:
        return self.long_term_source_weight if source == "long_term" else self.session_source_weight


class SessionMemory:
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

    def add_tool_result(self, message: dict[str, Any], *, tool_name: str, ok: bool, content: str) -> None:
        self._add(
            MemoryEntry(
                type=MemoryType.TOOL_RESULT,
                messages=[message],
                content=content,
                metadata={"tool": tool_name, "ok": ok},
            )
        )

    def messages(self, query: str | None = None) -> list[dict[str, Any]]:
        return [
            self.system_message,
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
        boundary = _recent_round_boundary(combined, retain_rounds)
        self.old_entries = combined[:boundary]
        self.active_entries = combined[boundary:]
        return len(self.old_entries)

    def move_entries_before_recent_rounds_to_old(self, retain_rounds: int) -> int:
        if retain_rounds < 0:
            raise ValueError("retain_rounds must be non-negative")
        boundary = _recent_round_boundary(self.active_entries, retain_rounds)
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

        if _entry_has_tool_calls(first):
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
                    content=_entry_content(entry),
                    created_at=entry.created_at,
                    score=0.0,
                )
                for entry in session_entries
                if _entry_content(entry)
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


class LongTermMemory:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.records: list[LongTermMemoryRecord] = []
        self._keys: set[str] = set()
        self._load()

    def save(self, content: str) -> LongTermMemorySaveResult:
        cleaned = content.strip()
        if not cleaned:
            raise ValueError("long-term memory content cannot be empty")

        key = _dedupe_key(cleaned)
        existing = self._find_by_key(key)
        if existing is not None:
            return LongTermMemorySaveResult(record=existing, saved=False)

        record = LongTermMemoryRecord(content=cleaned, created_at=_now())
        self.records.append(record)
        self._keys.add(key)
        self._write()
        return LongTermMemorySaveResult(record=record, saved=True)

    def _load(self) -> None:
        if not self.path.exists():
            return

        raw = json.loads(self.path.read_text(encoding="utf-8"))
        if isinstance(raw, dict):
            raw_records = raw.get("memories") or []
        elif isinstance(raw, list):
            raw_records = raw
        else:
            raw_records = []

        for item in raw_records:
            if not isinstance(item, dict):
                continue
            content = str(item.get("content") or "").strip()
            created_at = str(item.get("created_at") or "").strip()
            if not content:
                continue
            key = _dedupe_key(content)
            if key in self._keys:
                continue
            self.records.append(
                LongTermMemoryRecord(content=content, created_at=created_at or _now())
            )
            self._keys.add(key)

    def _write(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "memories": [
                {"content": record.content, "created_at": record.created_at}
                for record in self.records
            ]
        }
        self.path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _find_by_key(self, key: str) -> LongTermMemoryRecord | None:
        for record in self.records:
            if _dedupe_key(record.content) == key:
                return record
        return None


def default_long_term_memory_path(workspace: Path) -> Path:
    return workspace / ".mikucli" / "long_term_memory.json"


def token_usage_ratio(total_tokens: int | None, context_window_tokens: int) -> float | None:
    if total_tokens is None:
        return None
    if context_window_tokens <= 0:
        raise ValueError("context_window_tokens must be positive")
    return total_tokens / context_window_tokens


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _dedupe_key(content: str) -> str:
    return " ".join(content.casefold().split())


def _keywords(text: str) -> set[str]:
    return {
        term
        for term in re.findall(r"[A-Za-z0-9_]+", text.casefold())
        if len(term) > 2
    }


def _parse_timestamp(raw: str) -> datetime | None:
    try:
        value = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _chunk_memory_entries(entries: list[MemoryEntry], chunk_chars: int) -> list[str]:
    chunks: list[str] = []
    current: list[str] = []
    current_length = 0
    for entry in entries:
        text = _entry_text(entry)
        if not text:
            continue
        separator_length = 2 if current else 0
        if current and current_length + separator_length + len(text) > chunk_chars:
            chunks.append("\n\n".join(current))
            current = []
            current_length = 0
        current.append(text)
        current_length += separator_length + len(text)
    if current:
        chunks.append("\n\n".join(current))
    return chunks


def _entry_text(entry: MemoryEntry) -> str:
    text = _entry_content(entry)
    text = text.strip()
    if not text:
        return ""
    return f"[{entry.type.value}] {text}"


def _entry_content(entry: MemoryEntry) -> str:
    return (entry.content or _messages_text(entry.messages)).strip()


def _parse_json_string_list(raw: str) -> list[str]:
    cleaned = raw.strip()
    if not cleaned:
        return []
    start = cleaned.find("[")
    end = cleaned.rfind("]")
    if start != -1 and end != -1 and end > start:
        cleaned = cleaned[start : end + 1]
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        return []
    if not isinstance(parsed, list):
        return []
    return [item.strip() for item in parsed if isinstance(item, str) and item.strip()]


def _recent_round_boundary(entries: list[MemoryEntry], retain_rounds: int) -> int:
    if retain_rounds == 0:
        return len(entries)

    seen = 0
    for index in range(len(entries) - 1, -1, -1):
        if _is_real_user_turn(entries[index]):
            seen += 1
            if seen == retain_rounds:
                return index
    return 0


def _is_real_user_turn(entry: MemoryEntry) -> bool:
    if entry.type != MemoryType.CONVERSATION or not entry.messages:
        return False
    message = entry.messages[0]
    if message.get("role") != "user":
        return False
    content = str(message.get("content") or "")
    return not content.startswith("Tool result for ")


def _entry_has_tool_calls(entry: MemoryEntry) -> bool:
    return any(message.get("tool_calls") for message in entry.messages)


def _messages_text(messages: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for message in messages:
        role = str(message.get("role") or "message")
        content = str(message.get("content") or "")
        if content:
            parts.append(f"{role}: {content}")
    return "\n".join(parts)


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "\n... truncated ..."

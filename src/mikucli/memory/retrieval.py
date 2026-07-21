from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Callable, Literal

from .models import LongTermMemoryRecord, MemoryEntry, RetrievedMemory
from .utilities import entry_content, keywords, parse_timestamp


class MemoryRetriever:
    """Rank session and long-term memories for a task query."""

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
        query_terms = keywords(query)
        candidates: list[RetrievedMemory] = []
        for entry in session_entries:
            content = entry_content(entry)
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
                parse_timestamp(candidate.created_at),
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
        content_terms = keywords(content)
        if not content_terms:
            return 0.0
        return len(query_terms & content_terms) / len(query_terms)

    def time_decay(self, created_at: str) -> float:
        created = parse_timestamp(created_at)
        if created is None:
            return self.min_time_decay
        age = self.now() - created
        if age.total_seconds() <= 0:
            return 1.0
        ratio = min(age / self.time_decay_window, 1.0)
        return 1.0 - ((1.0 - self.min_time_decay) * ratio)

    def source_weight(self, source: Literal["session", "long_term"]) -> float:
        return self.long_term_source_weight if source == "long_term" else self.session_source_weight

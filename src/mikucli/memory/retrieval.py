from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
from threading import RLock
from typing import Callable, Literal

from ..codebase.embeddings import EmbeddingClient, document_inputs, query_inputs
from .models import LongTermMemoryRecord, MemoryEntry, RetrievedMemory
from .utilities import entry_content, keywords, parse_timestamp


class MemoryRetrievalError(RuntimeError):
    """Raised when mandatory semantic memory retrieval cannot be completed."""


class MemoryRetriever:
    """Rank session and long-term memories using keyword and semantic relevance."""

    def __init__(
        self,
        *,
        embedding_client: EmbeddingClient,
        embedding_model: str,
        now: Callable[[], datetime] | None = None,
        keyword_weight: float = 0.5,
        semantic_weight: float = 0.5,
        min_semantic_similarity: float = 0.5,
        long_term_source_weight: float = 1.2,
        session_source_weight: float = 1.0,
        time_decay_window: timedelta = timedelta(hours=24),
        min_time_decay: float = 0.5,
    ) -> None:
        if not embedding_model.strip():
            raise ValueError("embedding_model must not be empty")
        if keyword_weight < 0 or semantic_weight < 0:
            raise ValueError("memory retrieval weights must be non-negative")
        if not math.isclose(keyword_weight + semantic_weight, 1.0):
            raise ValueError("memory retrieval weights must add up to 1.0")
        if not 0.0 <= min_semantic_similarity <= 1.0:
            raise ValueError("min_semantic_similarity must be between 0.0 and 1.0")

        self.embedding_client = embedding_client
        self.embedding_model = embedding_model
        self.now = now or (lambda: datetime.now(timezone.utc))
        self.keyword_weight = keyword_weight
        self.semantic_weight = semantic_weight
        self.min_semantic_similarity = min_semantic_similarity
        self.long_term_source_weight = long_term_source_weight
        self.session_source_weight = session_source_weight
        self.time_decay_window = time_decay_window
        self.min_time_decay = min_time_decay
        self._query_embedding_cache: dict[str, list[float]] = {}
        self._content_embedding_cache: dict[str, list[float]] = {}
        self._cache_lock = RLock()

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

        raw_candidates: list[tuple[Literal["session", "long_term"], str, str]] = []
        raw_candidates.extend(
            ("session", content, entry.created_at)
            for entry in session_entries
            if (content := entry_content(entry))
        )
        raw_candidates.extend(
            ("long_term", record.content, record.created_at)
            for record in long_term_records
            if record.content
        )
        if not raw_candidates:
            return []

        query_embedding, content_embeddings = self._embeddings(
            query,
            [content for _, content, _ in raw_candidates],
        )
        query_terms = keywords(query)
        candidates: list[RetrievedMemory] = []
        for source, content, created_at in raw_candidates:
            semantic_similarity = self.semantic_similarity(
                query_embedding,
                content_embeddings[content],
            )
            score = self.score(
                query_terms=query_terms,
                content=content,
                semantic_similarity=semantic_similarity,
                created_at=created_at,
                source=source,
            )
            if score > 0:
                candidates.append(
                    RetrievedMemory(
                        source=source,
                        content=content,
                        created_at=created_at,
                        score=score,
                    )
                )
        return sorted(
            candidates,
            key=lambda candidate: (
                candidate.score,
                parse_timestamp(candidate.created_at)
                or datetime.min.replace(tzinfo=timezone.utc),
                1 if candidate.source == "long_term" else 0,
            ),
            reverse=True,
        )[:limit]

    def score(
        self,
        *,
        query_terms: set[str],
        content: str,
        semantic_similarity: float,
        created_at: str,
        source: Literal["session", "long_term"],
    ) -> float:
        keyword_score = self.keyword_matchup(query_terms, content)
        if keyword_score <= 0 and semantic_similarity < self.min_semantic_similarity:
            return 0.0
        bounded_semantic_score = min(max(semantic_similarity, 0.0), 1.0)
        relevance = (
            self.keyword_weight * keyword_score
            + self.semantic_weight * bounded_semantic_score
        )
        return relevance * self.time_decay(created_at) * self.source_weight(source)

    def keyword_matchup(self, query_terms: set[str], content: str) -> float:
        if not query_terms:
            return 0.0
        content_terms = keywords(content)
        if not content_terms:
            return 0.0
        return len(query_terms & content_terms) / len(query_terms)

    def semantic_similarity(self, query_embedding: list[float], content_embedding: list[float]) -> float:
        if len(query_embedding) != len(content_embedding):
            raise MemoryRetrievalError(
                "Memory embedding dimensions do not match; check the configured embedding model."
            )
        query_norm = math.sqrt(sum(value * value for value in query_embedding))
        content_norm = math.sqrt(sum(value * value for value in content_embedding))
        if query_norm == 0 or content_norm == 0:
            return 0.0
        return sum(
            query_value * content_value
            for query_value, content_value in zip(query_embedding, content_embedding, strict=True)
        ) / (query_norm * content_norm)

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

    def _embeddings(
        self,
        query: str,
        contents: list[str],
    ) -> tuple[list[float], dict[str, list[float]]]:
        unique_contents = list(dict.fromkeys(contents))
        with self._cache_lock:
            query_embedding = self._query_embedding_cache.get(query)
            cached_contents = {
                content: self._content_embedding_cache[content]
                for content in unique_contents
                if content in self._content_embedding_cache
            }

        missing_contents = [content for content in unique_contents if content not in cached_contents]
        inputs: list[str] = []
        if query_embedding is None:
            inputs.extend(query_inputs(self.embedding_model, [query]))
        inputs.extend(document_inputs(self.embedding_model, missing_contents))
        if inputs:
            embeddings = self.embedding_client.embed(inputs)
            if len(embeddings) != len(inputs):
                raise MemoryRetrievalError(
                    "Embedding service returned an unexpected number of memory embeddings."
                )
            offset = 0
            if query_embedding is None:
                query_embedding = embeddings[0]
                offset = 1
            for content, embedding in zip(
                missing_contents,
                embeddings[offset:],
                strict=True,
            ):
                cached_contents[content] = embedding
            if query_embedding is None:
                raise MemoryRetrievalError("Embedding service did not return a query embedding.")
            with self._cache_lock:
                self._query_embedding_cache.setdefault(query, query_embedding)
                for content in missing_contents:
                    self._content_embedding_cache.setdefault(content, cached_contents[content])

        if query_embedding is None:
            raise MemoryRetrievalError("Embedding service did not return a query embedding.")
        return query_embedding, cached_contents

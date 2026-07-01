from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class IndexedFile:
    path: str
    absolute_path: Path
    size: int
    mtime_ns: int
    content_hash: str


@dataclass(frozen=True)
class FileSkip:
    path: str
    reason: str


@dataclass(frozen=True)
class CodeChunk:
    path: str
    start_line: int
    end_line: int
    kind: str
    symbol: str
    content: str
    content_hash: str
    chunk_hash: str


@dataclass(frozen=True)
class SearchResult:
    path: str
    start_line: int
    end_line: int
    kind: str
    symbol: str
    content: str
    hybrid_score: float
    semantic_score: float | None = None
    lexical_score: float | None = None
    semantic_rank: int | None = None
    lexical_rank: int | None = None


@dataclass
class IndexStats:
    files_scanned: int = 0
    files_indexed: int = 0
    files_skipped: int = 0
    chunks_embedded: int = 0
    embedding_model: str = ""
    index_path: Path | None = None
    skips: list[FileSkip] = field(default_factory=list)

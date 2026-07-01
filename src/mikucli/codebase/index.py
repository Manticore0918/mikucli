from __future__ import annotations

import json
import math
import os
import re
import shutil
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

from .types import CodeChunk, IndexedFile, SearchResult


SCHEMA_VERSION = "1"
RRF_K = 60


class CodebaseIndexError(RuntimeError):
    pass


@dataclass(frozen=True)
class ActiveIndex:
    root: Path

    @property
    def db_path(self) -> Path:
        return self.root / "index.sqlite3"

    @property
    def tmp_dir(self) -> Path:
        return self.root / "tmp"

    def exists(self) -> bool:
        return self.db_path.exists()


class CodebaseIndexWriter:
    def __init__(self, active: ActiveIndex, embedding_model: str) -> None:
        self.active = active
        self.embedding_model = embedding_model
        self.temp_path = active.tmp_dir / f"index-{int(time.time() * 1000)}.sqlite3"
        self.conn: sqlite3.Connection | None = None

    def __enter__(self) -> "CodebaseIndexWriter":
        self.active.tmp_dir.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.temp_path)
        _init_schema(self.conn)
        _set_meta(self.conn, "schema_version", SCHEMA_VERSION)
        _set_meta(self.conn, "embedding_model", self.embedding_model)
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        if self.conn is not None:
            self.conn.close()
        if exc_type is not None and self.temp_path.exists():
            self.temp_path.unlink()

    def add_file(self, file: IndexedFile, chunks: list[CodeChunk], embeddings: list[list[float]]) -> None:
        if self.conn is None:
            raise CodebaseIndexError("index writer is not open")
        if len(chunks) != len(embeddings):
            raise CodebaseIndexError("chunk and embedding counts differ")
        self.conn.execute(
            """
            INSERT INTO files(path, size, mtime_ns, content_hash, chunk_count)
            VALUES (?, ?, ?, ?, ?)
            """,
            (file.path, file.size, file.mtime_ns, file.content_hash, len(chunks)),
        )
        for chunk, embedding in zip(chunks, embeddings):
            cursor = self.conn.execute(
                """
                INSERT INTO chunks(
                    path, start_line, end_line, kind, symbol, content,
                    content_hash, chunk_hash, embedding, embedding_norm
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    chunk.path,
                    chunk.start_line,
                    chunk.end_line,
                    chunk.kind,
                    chunk.symbol,
                    chunk.content,
                    chunk.content_hash,
                    chunk.chunk_hash,
                    json.dumps(embedding),
                    _norm(embedding),
                ),
            )
            rowid = cursor.lastrowid
            self.conn.execute(
                "INSERT INTO chunks_fts(rowid, path, symbol, content) VALUES (?, ?, ?, ?)",
                (rowid, chunk.path, chunk.symbol, chunk.content),
            )

    def commit_active(self) -> Path:
        if self.conn is None:
            raise CodebaseIndexError("index writer is not open")
        self.conn.commit()
        _validate_index(self.conn)
        self.conn.close()
        self.conn = None
        self.active.root.mkdir(parents=True, exist_ok=True)
        os.replace(self.temp_path, self.active.db_path)
        _cleanup_tmp(self.active.tmp_dir)
        return self.active.db_path


class CodebaseIndexReader:
    def __init__(self, active: ActiveIndex) -> None:
        self.active = active

    def search(self, query: str, query_embedding: list[float], limit: int = 8) -> list[SearchResult]:
        if not self.active.exists():
            raise CodebaseIndexError("No Codebase Index exists. Run /index first.")
        conn = sqlite3.connect(self.active.db_path)
        try:
            conn.row_factory = sqlite3.Row
            semantic = _semantic_results(conn, query_embedding, limit=max(limit * 4, 20))
            lexical = _lexical_results(conn, query, limit=max(limit * 4, 20))
            return _fuse_results(semantic, lexical, limit)
        finally:
            conn.close()


def _init_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE meta(
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );

        CREATE TABLE files(
            path TEXT PRIMARY KEY,
            size INTEGER NOT NULL,
            mtime_ns INTEGER NOT NULL,
            content_hash TEXT NOT NULL,
            chunk_count INTEGER NOT NULL
        );

        CREATE TABLE chunks(
            id INTEGER PRIMARY KEY,
            path TEXT NOT NULL,
            start_line INTEGER NOT NULL,
            end_line INTEGER NOT NULL,
            kind TEXT NOT NULL,
            symbol TEXT NOT NULL,
            content TEXT NOT NULL,
            content_hash TEXT NOT NULL,
            chunk_hash TEXT NOT NULL,
            embedding TEXT NOT NULL,
            embedding_norm REAL NOT NULL
        );

        CREATE VIRTUAL TABLE chunks_fts USING fts5(path, symbol, content);
        """
    )


def _set_meta(conn: sqlite3.Connection, key: str, value: str) -> None:
    conn.execute("INSERT OR REPLACE INTO meta(key, value) VALUES (?, ?)", (key, value))


def _validate_index(conn: sqlite3.Connection) -> None:
    schema_version = conn.execute("SELECT value FROM meta WHERE key = 'schema_version'").fetchone()
    if schema_version is None or schema_version[0] != SCHEMA_VERSION:
        raise CodebaseIndexError("Codebase Index validation failed: schema version missing")
    chunk_count = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    fts_count = conn.execute("SELECT COUNT(*) FROM chunks_fts").fetchone()[0]
    if chunk_count != fts_count:
        raise CodebaseIndexError("Codebase Index validation failed: FTS row count mismatch")


def _semantic_results(conn: sqlite3.Connection, query_embedding: list[float], limit: int) -> list[SearchResult]:
    query_norm = _norm(query_embedding)
    if query_norm == 0:
        return []
    results: list[SearchResult] = []
    for row in conn.execute("SELECT * FROM chunks"):
        embedding = json.loads(row["embedding"])
        score = _cosine(query_embedding, query_norm, embedding, float(row["embedding_norm"]))
        results.append(_row_to_result(row, hybrid=0.0, semantic=score, lexical=None))
    return sorted(results, key=lambda result: result.semantic_score or 0.0, reverse=True)[:limit]


def _lexical_results(conn: sqlite3.Connection, query: str, limit: int) -> list[SearchResult]:
    expression = _fts_query(query)
    if not expression:
        return []
    rows = conn.execute(
        """
        SELECT chunks.*, bm25(chunks_fts) AS bm25_score
        FROM chunks_fts
        JOIN chunks ON chunks.id = chunks_fts.rowid
        WHERE chunks_fts MATCH ?
        ORDER BY bm25_score
        LIMIT ?
        """,
        (expression, limit),
    ).fetchall()
    results: list[SearchResult] = []
    for row in rows:
        bm25_score = float(row["bm25_score"])
        results.append(_row_to_result(row, hybrid=0.0, semantic=None, lexical=bm25_score))
    return results


def _fuse_results(
    semantic: list[SearchResult],
    lexical: list[SearchResult],
    limit: int,
) -> list[SearchResult]:
    by_key: dict[tuple[str, int, int, str], dict[str, object]] = {}

    def add(result: SearchResult, rank: int, source: str) -> None:
        key = (result.path, result.start_line, result.end_line, result.content)
        item = by_key.setdefault(
            key,
            {
                "result": result,
                "hybrid": 0.0,
                "semantic_score": None,
                "lexical_score": None,
                "semantic_rank": None,
                "lexical_rank": None,
            },
        )
        item["hybrid"] = float(item["hybrid"]) + (1 / (RRF_K + rank))
        if source == "semantic":
            item["semantic_score"] = result.semantic_score
            item["semantic_rank"] = rank
        else:
            item["lexical_score"] = result.lexical_score
            item["lexical_rank"] = rank

    for index, result in enumerate(semantic, start=1):
        add(result, index, "semantic")
    for index, result in enumerate(lexical, start=1):
        add(result, index, "lexical")

    fused: list[SearchResult] = []
    for item in by_key.values():
        result = item["result"]
        assert isinstance(result, SearchResult)
        fused.append(
            SearchResult(
                path=result.path,
                start_line=result.start_line,
                end_line=result.end_line,
                kind=result.kind,
                symbol=result.symbol,
                content=result.content,
                hybrid_score=float(item["hybrid"]),
                semantic_score=item["semantic_score"],  # type: ignore[arg-type]
                lexical_score=item["lexical_score"],  # type: ignore[arg-type]
                semantic_rank=item["semantic_rank"],  # type: ignore[arg-type]
                lexical_rank=item["lexical_rank"],  # type: ignore[arg-type]
            )
        )
    return sorted(fused, key=lambda result: result.hybrid_score, reverse=True)[:limit]


def _row_to_result(
    row: sqlite3.Row,
    *,
    hybrid: float,
    semantic: float | None,
    lexical: float | None,
) -> SearchResult:
    return SearchResult(
        path=str(row["path"]),
        start_line=int(row["start_line"]),
        end_line=int(row["end_line"]),
        kind=str(row["kind"]),
        symbol=str(row["symbol"]),
        content=str(row["content"]),
        hybrid_score=hybrid,
        semantic_score=semantic,
        lexical_score=lexical,
    )


def _fts_query(query: str) -> str:
    terms = re.findall(r"[A-Za-z0-9_]+", query)
    return " OR ".join(f'"{term}"' for term in terms)


def _norm(values: list[float]) -> float:
    return math.sqrt(sum(value * value for value in values))


def _cosine(left: list[float], left_norm: float, right: list[float], right_norm: float) -> float:
    if left_norm == 0 or right_norm == 0:
        return 0.0
    size = min(len(left), len(right))
    return sum(left[index] * right[index] for index in range(size)) / (left_norm * right_norm)


def _cleanup_tmp(tmp_dir: Path) -> None:
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)

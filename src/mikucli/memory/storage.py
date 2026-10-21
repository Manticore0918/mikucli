from __future__ import annotations

import math
import os
import sqlite3
import tempfile
import time
from collections.abc import Iterator
from contextlib import closing, contextmanager
from pathlib import Path

from .migration import read_legacy_records
from .models import LongTermMemoryRecord, LongTermMemorySaveResult, utc_now
from .utilities import dedupe_key

APPLICATION_ID = 0x4D494B55
SCHEMA_VERSION = 1


class MemoryStorageError(RuntimeError):
    """Persistent memory could not be read, committed, or recovered."""


def restore_marker(path: Path) -> Path:
    return path.with_name(path.name + ".restore-pending")


def sync_file(path: Path) -> None:
    with path.open("r+b") as stream:
        os.fsync(stream.fileno())


def sync_directory(path: Path) -> None:
    if os.name != "nt":
        descriptor = os.open(path, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def validate_database(connection: sqlite3.Connection) -> None:
    if connection.execute("PRAGMA quick_check").fetchall() != [("ok",)]:
        raise MemoryStorageError("memory database integrity check failed")
    if connection.execute("PRAGMA application_id").fetchone()[0] != APPLICATION_ID:
        raise MemoryStorageError("not a mikucli memory database")
    if connection.execute("PRAGMA user_version").fetchone()[0] != SCHEMA_VERSION:
        raise MemoryStorageError("unsupported memory database schema version")
    rows = connection.execute(
        "SELECT normalized_key, content, created_at FROM memories"
    )
    for key, content, timestamp in rows:
        if (
            not isinstance(content, str)
            or not content.strip()
            or not isinstance(timestamp, str)
            or not timestamp.strip()
            or key != dedupe_key(content)
        ):
            raise MemoryStorageError("invalid record in memory database")
    if connection.execute(
        "SELECT value FROM metadata WHERE key = 'legacy_imported'"
    ).fetchone() != ("1",):
        raise MemoryStorageError("memory migration is incomplete")


def copy_database(source: Path, destination: Path, timeout: float) -> None:
    """Create and validate a consistent snapshot through SQLite's backup API."""
    uri = source.resolve().as_uri() + "?mode=ro"
    with closing(sqlite3.connect(uri, uri=True, timeout=timeout)) as reader:
        validate_database(reader)
        with closing(sqlite3.connect(destination, timeout=timeout)) as writer:
            writer.execute("PRAGMA synchronous=FULL")
            deadline = time.monotonic() + timeout

            def progress(status: int, remaining: int, total: int) -> None:
                if status != sqlite3.SQLITE_DONE and time.monotonic() >= deadline:
                    raise MemoryStorageError("memory backup timed out")

            reader.backup(writer, pages=128, progress=progress, sleep=0.05)
            validate_database(writer)
    sync_file(destination)


class SQLiteMemoryStore:
    def __init__(self, path: Path, legacy_path: Path, *, timeout: float = 5.0) -> None:
        self.path = path
        self.legacy_path = legacy_path
        if not math.isfinite(timeout) or timeout < 0:
            raise ValueError("memory lock timeout must be finite and non-negative")
        self.timeout = timeout
        self._initialized = False

    @contextmanager
    def connect(self, *, create: bool = False) -> Iterator[sqlite3.Connection]:
        if restore_marker(self.path).exists():
            raise MemoryStorageError(
                "interrupted memory restore; finish offline recovery with --resume"
            )
        mode = "rwc" if create else "rw"
        uri = self.path.resolve().as_uri() + f"?mode={mode}"
        with closing(
            sqlite3.connect(uri, uri=True, timeout=self.timeout, isolation_level=None)
        ) as connection:
            connection.execute("PRAGMA synchronous=FULL")
            try:
                yield connection
            finally:
                if connection.in_transaction:
                    connection.rollback()

    def initialize(self) -> bool:
        if (
            not self.path.exists()
            and self.path.with_name(self.path.name + ".bak").exists()
        ):
            raise MemoryStorageError(
                "memory database is missing; restore its existing backup"
            )
        parent_existed = self.path.parent.exists()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not parent_existed:
            sync_directory(self.path.parent.parent)
        migrated = False
        with self.connect(create=not self._initialized) as connection:
            connection.execute("BEGIN IMMEDIATE")
            version = connection.execute("PRAGMA user_version").fetchone()[0]
            application = connection.execute("PRAGMA application_id").fetchone()[0]
            tables = connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
            if version == 0 and application == 0 and not tables:
                migrated = True
                connection.execute(
                    "CREATE TABLE memories (normalized_key TEXT PRIMARY KEY NOT NULL, "
                    "content TEXT NOT NULL CHECK(length(trim(content)) > 0), "
                    "created_at TEXT NOT NULL CHECK(length(created_at) > 0))"
                )
                connection.execute(
                    "CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
                )
                connection.execute(f"PRAGMA application_id={APPLICATION_ID}")
                connection.execute(f"PRAGMA user_version={SCHEMA_VERSION}")
                for record in read_legacy_records(self.legacy_path):
                    connection.execute(
                        "INSERT INTO memories VALUES (?, ?, ?) ON CONFLICT(normalized_key) DO NOTHING",
                        (dedupe_key(record.content), record.content, record.created_at),
                    )
                connection.execute(
                    "INSERT INTO metadata VALUES ('legacy_imported', '1')"
                )
            validate_database(connection)
            connection.commit()
        sync_directory(self.path.parent)
        self._initialized = True
        return migrated

    def records(self) -> list[LongTermMemoryRecord]:
        with self.connect() as connection:
            connection.execute("BEGIN")
            validate_database(connection)
            return [
                LongTermMemoryRecord(*row)
                for row in connection.execute(
                    "SELECT content, created_at FROM memories ORDER BY rowid"
                )
            ]

    def save_many(self, contents: list[str]) -> list[LongTermMemorySaveResult]:
        results = []
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            validate_database(connection)
            for content in contents:
                key = dedupe_key(content)
                cursor = connection.execute(
                    "INSERT INTO memories VALUES (?, ?, ?) ON CONFLICT(normalized_key) DO NOTHING",
                    (key, content, utc_now()),
                )
                row = connection.execute(
                    "SELECT content, created_at FROM memories WHERE normalized_key = ?",
                    (key,),
                ).fetchone()
                results.append(
                    LongTermMemorySaveResult(
                        LongTermMemoryRecord(*row), saved=cursor.rowcount == 1
                    )
                )
            connection.commit()
        return results

    def backup(self, destination: Path) -> Path:
        reserved = {self.path, self.legacy_path, restore_marker(self.path)}
        reserved.update(
            self.path.with_name(self.path.name + suffix)
            for suffix in ("-journal", "-wal", "-shm")
        )
        if destination.resolve() in {path.resolve() for path in reserved}:
            raise MemoryStorageError(
                "backup must not overwrite memory storage or its recovery files"
            )
        destination.parent.mkdir(parents=True, exist_ok=True)
        descriptor, name = tempfile.mkstemp(
            prefix=destination.name + ".", suffix=".tmp", dir=destination.parent
        )
        os.close(descriptor)
        temporary = Path(name)
        try:
            # Serialize snapshot publication with every writer. A late backup can
            # never replace a newer snapshot with an older committed generation.
            with self.connect() as connection:
                connection.execute("BEGIN IMMEDIATE")
                copy_database(self.path, temporary, self.timeout)
                os.replace(temporary, destination)
                sync_directory(destination.parent)
            return destination
        finally:
            temporary.unlink(missing_ok=True)

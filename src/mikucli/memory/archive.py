"""Transactional session compression evidence, separate from durable facts."""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from threading import RLock
from uuid import uuid4

from .ledger import ProtectedRecord, canonical_entry, entry_checksum
from .models import MemoryEntry, MemoryType
from .storage import MemoryStorageError, sync_directory

APPLICATION_ID = 0x4D494B43
SCHEMA_VERSION = 1


class SessionArchive:
    """File-backed in the runner; path=None explicitly selects session-only storage."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = path
        self._lock = RLock()
        self._memory: sqlite3.Connection | None = None

    @contextmanager
    def _transaction(self):
        with self._lock:
            connection = None
            try:
                if self.path is None:
                    if self._memory is None:
                        self._memory = sqlite3.connect(
                            ":memory:", isolation_level=None, check_same_thread=False
                        )
                    connection = self._memory
                else:
                    self.path.parent.mkdir(parents=True, exist_ok=True)
                    connection = sqlite3.connect(
                        self.path, timeout=5, isolation_level=None
                    )
                connection.execute("PRAGMA synchronous=FULL")
                connection.execute("BEGIN IMMEDIATE")
                version = connection.execute("PRAGMA user_version").fetchone()[0]
                application = connection.execute("PRAGMA application_id").fetchone()[0]
                tables = connection.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
                if version == 0 and application == 0 and not tables:
                    connection.execute(
                        "CREATE TABLE sources (scope TEXT, id TEXT, payload TEXT NOT NULL, "
                        "checksum TEXT NOT NULL, PRIMARY KEY(scope, id))"
                    )
                    connection.execute(
                        "CREATE TABLE batches (scope TEXT, id TEXT, summary TEXT NOT NULL, "
                        "source_ids TEXT NOT NULL, ledger TEXT NOT NULL, PRIMARY KEY(scope, id))"
                    )
                    connection.execute(f"PRAGMA application_id={APPLICATION_ID}")
                    connection.execute(f"PRAGMA user_version={SCHEMA_VERSION}")
                elif version != SCHEMA_VERSION or application != APPLICATION_ID:
                    raise MemoryStorageError("Unsupported session archive database.")
                yield connection
                connection.commit()
                if self.path is not None:
                    sync_directory(self.path.parent)
            except (sqlite3.Error, OSError) as exc:
                raise MemoryStorageError(f"Session archive unavailable: {exc}") from exc
            finally:
                if connection is not None:
                    if connection.in_transaction:
                        connection.rollback()
                    if self.path is not None:
                        connection.close()

    def commit_batch(
        self,
        scope: str,
        sources: list[MemoryEntry],
        compressed_ids: list[str],
        summary: MemoryEntry,
        ledger: tuple[ProtectedRecord, ...],
    ) -> str:
        batch_id = uuid4().hex
        with self._transaction() as connection:
            for entry in sources:
                payload, checksum = canonical_entry(entry), entry_checksum(entry)
                previous = connection.execute(
                    "SELECT payload, checksum FROM sources WHERE scope=? AND id=?",
                    (scope, entry.entry_id),
                ).fetchone()
                if previous is not None and previous != (payload, checksum):
                    raise MemoryStorageError("An archived source entry was altered.")
                connection.execute(
                    "INSERT OR IGNORE INTO sources VALUES (?, ?, ?, ?)",
                    (scope, entry.entry_id, payload, checksum),
                )
            source_ids = {
                row[0]
                for row in connection.execute(
                    "SELECT id FROM sources WHERE scope=?", (scope,)
                )
            }
            required = set(compressed_ids) | {
                record.source_entry_id for record in ledger
            }
            required |= {
                record.status_source_id for record in ledger if record.status_source_id
            }
            if not required <= source_ids:
                raise MemoryStorageError(
                    "Compression references missing archive evidence."
                )
            connection.execute(
                "INSERT INTO batches VALUES (?, ?, ?, ?, ?)",
                (
                    scope,
                    batch_id,
                    canonical_entry(summary),
                    json.dumps(compressed_ids),
                    json.dumps(
                        [asdict(record) for record in ledger], ensure_ascii=False
                    ),
                ),
            )
        return batch_id

    def get_source(self, scope: str, entry_id: str) -> MemoryEntry:
        with self._transaction() as connection:
            row = connection.execute(
                "SELECT payload, checksum FROM sources WHERE scope=? AND id=?",
                (scope, entry_id),
            ).fetchone()
            if row is None:
                raise KeyError("Source entry not found in this session scope.")
            payload = json.loads(row[0])
            payload["type"] = MemoryType(payload["type"])
            entry = MemoryEntry(**payload)
            if entry_checksum(entry) != row[1]:
                raise MemoryStorageError("Archived source checksum mismatch.")
            return entry

    def batches(self, scope: str, *, include_prepared: bool = False) -> list[dict]:
        with self._transaction() as connection:
            rows = [
                {
                    "batch_id": row[0],
                    "summary": json.loads(row[1]),
                    "source_ids": json.loads(row[2]),
                    "ledger": json.loads(row[3]),
                }
                for row in connection.execute(
                    "SELECT id, summary, source_ids, ledger FROM batches WHERE scope=? ORDER BY rowid",
                    (scope,),
                )
            ]
            return [
                row
                for row in rows
                if include_prepared
                or row["summary"].get("metadata", {}).get("kind") != "compression_input"
            ]

    def sessions(self) -> list[dict]:
        with self._transaction() as connection:
            sessions: dict[str, dict] = {}
            for scope, summary in connection.execute(
                "SELECT scope, summary FROM batches ORDER BY rowid"
            ):
                info = sessions.setdefault(
                    scope, {"scope_id": scope, "batch_count": 0, "input_checkpoints": 0}
                )
                field = (
                    "input_checkpoints"
                    if json.loads(summary).get("metadata", {}).get("kind")
                    == "compression_input"
                    else "batch_count"
                )
                info[field] += 1
            return list(sessions.values())

    def close(self) -> None:
        if self._memory is not None:
            self._memory.close()
            self._memory = None

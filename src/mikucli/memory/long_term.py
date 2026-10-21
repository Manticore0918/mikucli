from __future__ import annotations

import sqlite3
import warnings
from collections.abc import Callable
from pathlib import Path

from .models import LongTermMemoryRecord, LongTermMemorySaveResult
from .storage import MemoryStorageError, SQLiteMemoryStore


class MemoryStorageWarning(RuntimeWarning):
    """Memory is unavailable, or its automatic backup could not be updated."""


class LongTermMemory:
    """Store workspace facts transactionally and read fresh committed snapshots."""

    def __init__(
        self,
        path: Path,
        *,
        timeout: float = 5.0,
        warning_callback: Callable[[str], None] | None = None,
    ) -> None:
        # Accept the former constructor path without ever overwriting its JSON.
        self.path = path.with_suffix(".sqlite3") if path.suffix == ".json" else path
        self.legacy_path = path if path.suffix == ".json" else path.with_suffix(".json")
        self.backup_path = self.path.with_name(self.path.name + ".bak")
        self._store = SQLiteMemoryStore(self.path, self.legacy_path, timeout=timeout)
        self._warning_callback = warning_callback
        self._last_warning = ""
        self.available = False
        self.error: str | None = None
        try:
            self._ensure_ready()
        except MemoryStorageError:
            pass  # The warning explains why persistent memory is disabled.

    @property
    def records(self) -> list[LongTermMemoryRecord]:
        try:
            self._ensure_ready()
            return self._store.records()
        except (sqlite3.Error, OSError, MemoryStorageError) as exc:
            self._unavailable(exc)
            return []

    def save(self, content: str) -> LongTermMemorySaveResult:
        return self.save_many([content])[0]

    def save_many(self, contents: list[str]) -> list[LongTermMemorySaveResult]:
        cleaned = [content.strip() for content in contents]
        if any(not content for content in cleaned):
            raise ValueError("long-term memory content cannot be empty")
        if not cleaned:
            return []
        try:
            self._ensure_ready()
            results = self._store.save_many(cleaned)
        except (sqlite3.Error, OSError, MemoryStorageError) as exc:
            self._unavailable(exc)
            raise MemoryStorageError(self.error) from exc
        if any(result.saved for result in results):
            self._automatic_backup()
        return results

    def backup(self, destination: Path | None = None) -> Path:
        try:
            self._ensure_ready()
            return self._store.backup(destination or self.backup_path)
        except (sqlite3.Error, OSError, MemoryStorageError) as exc:
            raise MemoryStorageError(f"memory backup failed: {exc}") from exc

    def _ensure_ready(self) -> None:
        if self.available:
            return
        try:
            migrated = self._store.initialize()
        except (
            sqlite3.Error,
            OSError,
            TypeError,
            ValueError,
            MemoryStorageError,
        ) as exc:
            self._unavailable(exc)
            raise MemoryStorageError(self.error) from exc
        self.available = True
        self.error = None
        self._last_warning = ""
        if migrated or not self.backup_path.exists():
            self._automatic_backup()

    def _automatic_backup(self) -> None:
        try:
            self._store.backup(self.backup_path)
        except (sqlite3.Error, OSError, MemoryStorageError) as exc:
            # The primary transaction has committed. Never turn a failed backup
            # into a misleading report that a durable save did not take place.
            self._warn(f"Memory committed, but backup could not be updated: {exc}")

    def _unavailable(self, exc: Exception) -> None:
        self.available = False
        message = str(exc)
        self.error = (
            message
            if message.startswith("Persistent memory unavailable")
            else (
                f"Persistent memory unavailable at {self.path}: {message}. "
                "Existing files are preserved; saves will fail until storage is repaired. "
                "Use python -m mikucli.memory restore --help for offline recovery."
            )
        )
        self._warn(self.error)

    def _warn(self, message: str) -> None:
        if message == self._last_warning:
            return
        self._last_warning = message
        if self._warning_callback is not None:
            self._warning_callback(message)
        else:
            warnings.warn(message, MemoryStorageWarning, stacklevel=3)


def default_long_term_memory_path(workspace: Path) -> Path:
    return workspace / ".mikucli" / "long_term_memory.sqlite3"

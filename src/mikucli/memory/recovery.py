from __future__ import annotations

import math
import os
import shutil
import sqlite3
import tempfile
from contextlib import closing
from pathlib import Path
from uuid import uuid4

from .storage import (
    MemoryStorageError,
    copy_database,
    restore_marker,
    sync_directory,
    sync_file,
)


def _check_not_busy(path: Path, timeout: float) -> None:
    if not path.exists():
        return
    try:
        with closing(
            sqlite3.connect(
                path.resolve().as_uri() + "?mode=rw", uri=True, timeout=timeout
            )
        ) as connection:
            connection.execute("BEGIN EXCLUSIVE")
            connection.rollback()
    except sqlite3.DatabaseError as exc:
        error_code = getattr(exc, "sqlite_errorcode", 0)
        if error_code & 0xFF not in {sqlite3.SQLITE_CORRUPT, sqlite3.SQLITE_NOTADB}:
            raise


def restore_memory(
    path: Path,
    backup: Path | None = None,
    *,
    resume: bool = False,
    timeout: float = 5.0,
) -> Path:
    """Restore offline, preserving the prior database and any recovery journals.

    All agent sessions must be stopped first. The marker prevents a new session
    from opening an incomplete restore, including after this process is killed.
    """
    path = path.expanduser().resolve()
    if not math.isfinite(timeout) or timeout < 0:
        raise ValueError("memory lock timeout must be finite and non-negative")
    backup = (backup or path.with_name(path.name + ".bak")).expanduser().resolve()
    marker = restore_marker(path)
    if backup in {path, path.with_suffix(".json")}:
        raise MemoryStorageError("restore source must be a separate memory backup")
    if marker.exists() and not resume:
        raise MemoryStorageError(
            "interrupted restore found; stop all sessions and rerun restore with --resume"
        )
    if not backup.is_file():
        raise MemoryStorageError(f"memory backup does not exist: {backup}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=path.name + ".restore-", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    staged = Path(name)
    try:
        # Validate and flush a complete replacement before touching the original.
        copy_database(backup, staged, timeout)
        _check_not_busy(path, timeout)
        archive = path.with_name(path.name + ".before-restore-" + uuid4().hex)
        archive.mkdir()
        originals = [
            path,
            *(
                path.with_name(path.name + suffix)
                for suffix in ("-journal", "-wal", "-shm")
            ),
        ]
        for original in [*originals, marker]:
            if original.exists():
                copied = archive / original.name
                shutil.copy2(original, copied)
                sync_file(copied)
        sync_directory(archive)
        # Exclusive creation also prevents two ordinary restores from running at
        # once. --resume is an explicit offline takeover of an interrupted one.
        with marker.open("w" if resume else "x", encoding="utf-8") as stream:
            stream.write(str(archive) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        sync_directory(path.parent)
        for journal in originals[1:]:
            if journal.exists():
                journal.replace(archive / journal.name)
        os.replace(staged, path)
        sync_directory(path.parent)
        marker.unlink()
        sync_directory(path.parent)
        return archive
    except (sqlite3.Error, OSError, MemoryStorageError) as exc:
        raise MemoryStorageError(
            f"memory restore failed; original files are preserved: {exc}"
        ) from exc
    finally:
        staged.unlink(missing_ok=True)

"""Offline memory backup and recovery, independent of model configuration."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from .archive import SessionArchive
from .long_term import LongTermMemory, default_long_term_memory_path
from .recovery import restore_memory
from .storage import MemoryStorageError


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    backup = commands.add_parser("backup", help="Write a verified memory snapshot.")
    backup.add_argument("--workspace", type=Path, default=Path.cwd())
    backup.add_argument(
        "--output",
        type=Path,
        help="Snapshot path; defaults to the automatic .sqlite3.bak file.",
    )
    restore = commands.add_parser(
        "restore", help="Restore memory; close all mikucli sessions first."
    )
    restore.add_argument("--workspace", type=Path, default=Path.cwd())
    restore.add_argument(
        "--backup",
        type=Path,
        help="Snapshot to restore; defaults to the automatic backup.",
    )
    restore.add_argument(
        "--resume",
        action="store_true",
        help="Continue an interrupted restore after stopping all sessions.",
    )
    sessions = commands.add_parser("sessions", help="List archived session scopes or their compression batches.")
    sessions.add_argument("--workspace", type=Path, default=Path.cwd())
    sessions.add_argument("--scope-id", help="Show batches and source IDs for this session scope.")
    source = commands.add_parser("source", help="Read a checksummed original session source without an API key.")
    source.add_argument("--workspace", type=Path, default=Path.cwd())
    source.add_argument("--scope-id", required=True)
    source.add_argument("--entry-id", required=True)
    args = parser.parse_args(argv)
    path = default_long_term_memory_path(args.workspace.expanduser().resolve())
    try:
        if args.command in {"sessions", "source"}:
            archive = SessionArchive(path.parent / "session_memory.sqlite3")
            if args.command == "source":
                result = asdict(archive.get_source(args.scope_id, args.entry_id))
            else:
                result = archive.batches(args.scope_id, include_prepared=True) if args.scope_id else archive.sessions()
            print(json.dumps(result, ensure_ascii=False, indent=2))
        elif args.command == "backup":
            memory = LongTermMemory(
                path, warning_callback=lambda message: print(message, file=sys.stderr)
            )
            print(f"Memory backup: {memory.backup(args.output)}")
        else:
            archive = restore_memory(path, args.backup, resume=args.resume)
            print(f"Memory restored: {path}\nPrevious files preserved: {archive}")
    except (MemoryStorageError, KeyError, ValueError) as exc:
        print(f"mikucli: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

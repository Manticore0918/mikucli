"""Offline memory backup and recovery, independent of model configuration."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

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
    args = parser.parse_args(argv)
    path = default_long_term_memory_path(args.workspace.expanduser().resolve())
    try:
        if args.command == "backup":
            memory = LongTermMemory(
                path, warning_callback=lambda message: print(message, file=sys.stderr)
            )
            print(f"Memory backup: {memory.backup(args.output)}")
        else:
            archive = restore_memory(path, args.backup, resume=args.resume)
            print(f"Memory restored: {path}\nPrevious files preserved: {archive}")
    except MemoryStorageError as exc:
        print(f"mikucli: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

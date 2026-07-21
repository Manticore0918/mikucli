from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any


MAX_READ_LINES = 400
MAX_READ_CHARS = 16_000


def is_hidden_internal(path: Path, workspace_root: Path) -> bool:
    try:
        parts = path.relative_to(workspace_root).parts
    except ValueError:
        return False
    return ".git" in parts or ".mikucli" in parts


def optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def validate_read_range(first: int, last: int, total_lines: int) -> str:
    if first < 1:
        return "start_line must be at least 1."
    if last < first:
        return "end_line must be greater than or equal to start_line."
    if total_lines == 0:
        return "cannot select a line range from an empty file."
    if first > total_lines:
        return f"start_line {first} is beyond the end of the file ({total_lines} lines)."
    if last > total_lines:
        return f"end_line {last} is beyond the end of the file ({total_lines} lines)."
    return ""


def large_file_message(path: str, *, total_lines: int, total_chars: int) -> str:
    return (
        "file is too large for an unbounded read.\n"
        f"Path: {path}\n"
        f"Lines: {total_lines}\n"
        f"Characters: {total_chars}\n"
        f"Maximum per read: {MAX_READ_LINES} lines and {MAX_READ_CHARS} characters.\n"
        "Use search_codebase to locate relevant passages when available, then call read_file again "
        "with 1-based inclusive start_line and end_line values that you choose."
    )


def extract_leading_env_assignments(command: str) -> tuple[str, dict[str, str]]:
    env: dict[str, str] = {}
    remaining = command.strip()
    while True:
        match = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)=([^\s]+)\s+(.+)$", remaining, flags=re.DOTALL)
        if match is None:
            break
        env[match.group(1)] = match.group(2)
        remaining = match.group(3).strip()
    return remaining, env


def shell_env(workspace_root: Path, command_env: dict[str, str]) -> dict[str, str]:
    env = os.environ.copy()
    src_path = workspace_root / "src"
    if src_path.is_dir():
        _prepend_env_path(env, "PYTHONPATH", str(src_path))
    for name, value in command_env.items():
        if name == "PYTHONPATH":
            _prepend_env_path(env, name, _normalize_pythonpath(value, workspace_root))
        else:
            env[name] = value
    return env


def _prepend_env_path(env: dict[str, str], name: str, value: str) -> None:
    current = env.get(name)
    env[name] = value if not current else value + os.pathsep + current


def _normalize_pythonpath(value: str, workspace_root: Path) -> str:
    parts = value.split(os.pathsep)
    normalized: list[str] = []
    for part in parts:
        if not part:
            continue
        path = Path(part)
        normalized.append(str(path if path.is_absolute() else workspace_root / path))
    return os.pathsep.join(normalized)

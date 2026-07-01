from __future__ import annotations

import fnmatch
import hashlib
from dataclasses import dataclass
from pathlib import Path

from .types import FileSkip, IndexedFile


INCLUDE_DIRS = ("src", "tests")
INCLUDE_PREFIXES = ("demo/src",)
BUILD_CONFIG_NAMES = {"README.md", "CONTEXT.md", "pyproject.toml", "pom.xml"}
TEXT_SUFFIXES = {
    ".cfg",
    ".ini",
    ".java",
    ".json",
    ".md",
    ".properties",
    ".py",
    ".toml",
    ".txt",
    ".xml",
    ".yaml",
    ".yml",
}
EXCLUDED_PARTS = {
    ".agents",
    ".git",
    ".mikucli",
    ".pytest_cache",
    ".venv",
    "__pycache__",
    "build",
    "dist",
    "target",
    "venv",
}
MAX_FILE_BYTES = 1_000_000


@dataclass(frozen=True)
class FileSelection:
    files: list[IndexedFile]
    skips: list[FileSkip]
    scanned: int


def select_index_files(workspace: Path, max_file_bytes: int = MAX_FILE_BYTES) -> FileSelection:
    root = workspace.resolve()
    gitignore = _GitIgnore(root)
    files: list[IndexedFile] = []
    skips: list[FileSkip] = []
    scanned = 0

    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        scanned += 1
        rel = path.relative_to(root).as_posix()
        reason = _skip_reason(root, path, rel, gitignore, max_file_bytes)
        if reason:
            skips.append(FileSkip(rel, reason))
            continue
        stat = path.stat()
        content = path.read_bytes()
        files.append(
            IndexedFile(
                path=rel,
                absolute_path=path,
                size=stat.st_size,
                mtime_ns=stat.st_mtime_ns,
                content_hash=hashlib.sha256(content).hexdigest(),
            )
        )

    return FileSelection(files=files, skips=skips, scanned=scanned)


def _skip_reason(
    root: Path,
    path: Path,
    rel: str,
    gitignore: "_GitIgnore",
    max_file_bytes: int,
) -> str:
    parts = set(path.relative_to(root).parts)
    if parts & EXCLUDED_PARTS:
        return "excluded path"
    if any(part.endswith(".egg-info") for part in parts):
        return "excluded package metadata"
    if path.name == ".env":
        return "excluded secret file"
    if path.suffix in {".pyc", ".pyo", ".pyd"}:
        return "excluded compiled Python file"
    if gitignore.matches(rel):
        return "ignored by .gitignore"
    if not _is_included(rel, path):
        return "outside configured include set"
    if path.suffix and path.suffix.lower() not in TEXT_SUFFIXES:
        return "unsupported file type"
    if path.stat().st_size > max_file_bytes:
        return "oversized file"
    sample = path.read_bytes()[:4096]
    if b"\0" in sample:
        return "binary file"
    try:
        path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return "non-UTF-8 text"
    return ""


def _is_included(rel: str, path: Path) -> bool:
    if path.name in BUILD_CONFIG_NAMES:
        return True
    first = rel.split("/", 1)[0]
    if first in INCLUDE_DIRS:
        return True
    return any(rel == prefix or rel.startswith(prefix + "/") for prefix in INCLUDE_PREFIXES)


class _GitIgnore:
    def __init__(self, root: Path) -> None:
        self.patterns = _read_gitignore(root / ".gitignore")

    def matches(self, rel: str) -> bool:
        for pattern in self.patterns:
            if _matches_pattern(rel, pattern):
                return True
        return False


def _read_gitignore(path: Path) -> list[str]:
    if not path.exists():
        return []
    patterns: list[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or line.startswith("!"):
            continue
        patterns.append(line)
    return patterns


def _matches_pattern(rel: str, pattern: str) -> bool:
    normalized = pattern.strip("/")
    if pattern.endswith("/"):
        return rel == normalized or rel.startswith(normalized + "/") or f"/{normalized}/" in f"/{rel}/"
    return fnmatch.fnmatch(rel, normalized) or fnmatch.fnmatch(Path(rel).name, normalized)

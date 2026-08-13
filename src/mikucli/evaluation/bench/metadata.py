from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable

from .models import BenchmarkCase
from .sandbox import sandbox_capabilities


REPORT_SCHEMA_VERSION = 2
_SUITE_SOURCE_FILES = (
    "__init__.py",
    "__main__.py",
    "adapters.py",
    "checks.py",
    "errors.py",
    "lifecycle.py",
    "metrics.py",
    "models.py",
    "persistence.py",
    "reporting.py",
    "runner.py",
    "sandbox.py",
    "sandbox_process.py",
    "support.py",
    "tasks.py",
)


def build_run_metadata(
    *,
    root: Path,
    model: str,
    max_steps: int,
    context_window_tokens: int,
    cases: Iterable[BenchmarkCase],
    repeat_count: int,
    shuffle_seed: int | None,
    client_type: str,
    memory_retriever_type: str | None,
    price: dict[str, float | None] | None,
) -> dict[str, Any]:
    selected = list(cases)
    suite = {
        "fingerprint": suite_fingerprint(),
        "case_ids": [case.id for case in selected],
    }
    configuration = {
        "model": model,
        "max_steps": max_steps,
        "context_window_tokens": context_window_tokens,
        "repeat_count": repeat_count,
        "shuffle_seed": shuffle_seed,
        "client_type": client_type,
        "memory_retriever_type": memory_retriever_type,
        "price": price,
        "sandbox": sandbox_capabilities(),
    }
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "suite": suite,
        "configuration": configuration,
        "configuration_fingerprint": _fingerprint(configuration),
        "environment": environment_metadata(root),
    }


def suite_fingerprint() -> str:
    package_root = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for name in _SUITE_SOURCE_FILES:
        path = package_root / name
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def environment_metadata(root: Path) -> dict[str, Any]:
    revision = _git_output(root, "rev-parse", "HEAD")
    status = _git_output(root, "status", "--porcelain")
    return {
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "executable": sys.executable,
        "git_revision": revision or None,
        "git_dirty": bool(status) if status is not None else None,
    }


def _fingerprint(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _git_output(root: Path, *arguments: str) -> str | None:
    try:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=root,
            stdin=subprocess.DEVNULL,
            text=True,
            capture_output=True,
            timeout=3,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()

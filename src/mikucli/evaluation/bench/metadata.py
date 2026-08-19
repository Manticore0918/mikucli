from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from datetime import timedelta
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlsplit, urlunsplit

from ..report_validation import REPORT_SCHEMA_VERSION
from .models import BenchmarkCase
from .sandbox import sandbox_capabilities


_SUITE_SOURCE_FILES = (
    "__init__.py",
    "__main__.py",
    "adapters.py",
    "catalog.py",
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
    "../report_validation.py",
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
    client: object,
    memory_retriever: object | None,
    price: dict[str, float | None] | None,
    runtime: str = "native",
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
        "single_agent_runtime": runtime,
        "client": client_metadata(client),
        "memory_retriever": memory_retriever_metadata(memory_retriever),
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


def client_metadata(client: object) -> dict[str, Any]:
    metadata: dict[str, Any] = {"type": _type_name(client)}
    base_url = getattr(client, "base_url", None)
    if isinstance(base_url, str):
        metadata["base_url"] = _safe_endpoint(base_url)
        query = urlsplit(base_url).query
        if query:
            metadata["base_url_query_fingerprint"] = hashlib.sha256(
                query.encode("utf-8")
            ).hexdigest()
    return metadata


def memory_retriever_metadata(retriever: object | None) -> dict[str, Any] | None:
    if retriever is None:
        return None
    metadata: dict[str, Any] = {"type": _type_name(retriever)}
    for name in (
        "embedding_model",
        "keyword_weight",
        "semantic_weight",
        "min_semantic_similarity",
        "long_term_source_weight",
        "session_source_weight",
        "min_time_decay",
    ):
        value = getattr(retriever, name, None)
        if isinstance(value, (str, int, float, bool)) or value is None:
            metadata[name] = value
    window = getattr(retriever, "time_decay_window", None)
    if isinstance(window, timedelta):
        metadata["time_decay_window_seconds"] = window.total_seconds()
    embedding_client = getattr(retriever, "embedding_client", None)
    if embedding_client is not None:
        embedding = {"type": _type_name(embedding_client)}
        model = getattr(embedding_client, "model", None)
        if isinstance(model, str):
            embedding["model"] = model
        base_url = getattr(embedding_client, "base_url", None)
        if isinstance(base_url, str):
            embedding["base_url"] = _safe_endpoint(base_url)
        metadata["embedding_client"] = embedding
    return metadata


def _fingerprint(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _safe_endpoint(value: str) -> str:
    parts = urlsplit(value)
    hostname = parts.hostname or ""
    if parts.port is not None:
        hostname = f"{hostname}:{parts.port}"
    return urlunsplit((parts.scheme, hostname, parts.path, "", ""))


def _type_name(value: object) -> str:
    cls = type(value)
    return f"{cls.__module__}.{cls.__qualname__}"


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

from __future__ import annotations

import argparse
import asyncio
from typing import Any

from mikucli.http_requests import request_json


def positive_int(value: str) -> int:
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return result


def add_runtime_arguments(parser: argparse.ArgumentParser) -> None:
    for flag, value_type, help_text in (
        ("request-timeout", float, "Per-request deadline in seconds (default 120)."),
        ("max-retries", int, "Transient HTTP retries, 0-10 (default 2)."),
        ("retry-max-delay", float, "Maximum retry delay in seconds (default 30)."),
        ("max-duration", float, "Overall task deadline in seconds (default 900)."),
        ("max-tokens", positive_int, "Stop at this cumulative reported token usage."),
        (
            "max-spend",
            float,
            "Stop at this estimated spend; requires both token prices.",
        ),
        (
            "max-output-bytes",
            positive_int,
            "Maximum tool output bytes per task (default 1048576).",
        ),
        (
            "prompt-token-price-per-million",
            float,
            "Prompt token price used for spend limits.",
        ),
        (
            "completion-token-price-per-million",
            float,
            "Completion token price used for spend limits.",
        ),
    ):
        parser.add_argument("--" + flag, type=value_type, default=None, help=help_text)
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit one JSON task result; tool approvals are denied.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check configuration, storage, and Ollama without a model chat call.",
    )
    parser.add_argument(
        "--team",
        action="store_true",
        help="Start in multi-agent mode, including for a one-shot task.",
    )


def error_result(
    error: BaseException, reason: str = "runtime_error", *, stopped: bool = False
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "status": "stopped" if stopped else "failed",
        "reason": reason,
        "final_answer": str(error),
        "log_path": None,
        "changed_paths": [],
        "metrics": {},
        "error": {"type": type(error).__name__, "message": str(error)},
    }


def dependency_check(config: Any, memory: Any) -> dict[str, Any]:
    checks = [
        {
            "name": "provider_configuration",
            "ok": True,
            "detail": "Credentials and HTTP endpoint configured; provider availability was not tested.",
        },
        {
            "name": "persistent_memory",
            "ok": memory.available,
            "detail": memory.error or "SQLite memory is available.",
        },
    ]
    try:
        data, _ = asyncio.run(
            request_json(
                config.ollama_base_url + "/api/tags",
                method="GET",
                policy=config.request_policy,
            )
        )
        names = {item["name"] for item in data["models"]}
        available = (
            config.embedding_model in names
            or config.embedding_model + ":latest" in names
        )
        detail = (
            "Embedding model is installed."
            if available
            else f"Run: ollama pull {config.embedding_model}"
        )
        checks.append({"name": "embedding_service", "ok": available, "detail": detail})
    except (RuntimeError, KeyError, TypeError, ValueError) as exc:
        checks.append(
            {
                "name": "embedding_service",
                "ok": False,
                "detail": f"Start Ollama and check MIKUCLI_OLLAMA_BASE_URL: {exc}",
            }
        )
    ok = all(item["ok"] for item in checks)
    return {
        "schema_version": 1,
        "status": "succeeded" if ok else "failed",
        "reason": "dependencies_ready" if ok else "dependency_unavailable",
        "checks": checks,
    }

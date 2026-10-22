"""Per-case evidence and a small allowlisted artifact safe for CI publication."""

from __future__ import annotations

import math
import re
import statistics
from collections import Counter
from typing import Any

from .metrics import _percentile, _wilson_interval

EVIDENCE_KIND = "mikucli-eval-evidence"
IDENTIFIER = re.compile(r"^[A-Za-z0-9_.:/+\-]{1,180}$")
CRITICAL_CATEGORIES = {
    "safety",
    "scope",
    "runtime_outcome",
    "reliability_contract",
    "tool_correctness",
}


def identifier(value: Any) -> str:
    return (
        value if isinstance(value, str) and IDENTIFIER.fullmatch(value) else "unknown"
    )


def number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        numeric = float(value)
    except OverflowError:
        return None
    return numeric if math.isfinite(numeric) and numeric >= 0 else None


def median_complete(values: list[Any]) -> float | None:
    checked = [number(value) for value in values]
    return statistics.median(checked) if checked and None not in checked else None


def summarize_trials(trials: list[dict[str, Any]]) -> dict[str, Any]:
    passed = sum(trial["passed"] is True for trial in trials)
    low, high = _wilson_interval(passed, len(trials))
    elapsed = [trial["elapsed_seconds"] for trial in trials]
    return {
        "trials": len(trials),
        "passed": passed,
        "pass_rate": passed / len(trials),
        "ci95_low": low,
        "ci95_high": high,
        "flaky": 0 < passed < len(trials),
        "median_latency_seconds": statistics.median(elapsed),
        "p95_latency_seconds": _percentile(elapsed, 0.95),
        "median_tokens": median_complete([trial["total_tokens"] for trial in trials]),
        "median_spend": median_complete([trial["estimated_spend"] for trial in trials]),
        "median_http_attempts": median_complete(
            [trial.get("http_attempts") for trial in trials]
        ),
        "median_model_calls": median_complete(
            [trial.get("model_calls") for trial in trials]
        ),
        "failure_categories": dict(
            Counter(
                category for trial in trials for category in trial["failure_categories"]
            )
        ),
        "critical_failures": sum(len(trial["critical_categories"]) for trial in trials),
        "outcome_reasons": dict(Counter(trial["outcome_reason"] for trial in trials)),
    }


def export_evidence(reports: list[dict[str, Any]]) -> dict[str, Any]:
    """Never copy prompts, answers, paths, arguments, endpoints, diffs, or messages."""
    exported = []
    for report in reports:
        trials_by_case: dict[str, list[dict[str, Any]]] = {}
        for result in report["results"]:
            metrics = result["metrics"]
            cost = metrics.get("cost") or {}
            checks = [*result["check_results"], *result["tool_correctness_results"]]
            failed = {
                identifier(check.get("category"))
                for check in checks
                if check.get("passed") is False
            }
            failures = {
                identifier(reason.get("category"))
                for reason in result["failure_reasons"]
            }
            if failures & {"cleanup_failed", "exception"}:
                failed.add("reliability_contract")
            spend = metrics.get("estimated_spend") or {}
            trial = {
                "trial_index": result["trial_index"],
                "passed": result["passed"],
                "outcome_status": identifier(result.get("outcome_status")),
                "outcome_reason": identifier(result.get("outcome_reason")),
                "elapsed_seconds": metrics["elapsed_seconds"],
                "prompt_tokens": cost.get("prompt_tokens"),
                "completion_tokens": cost.get("completion_tokens"),
                "total_tokens": cost.get("total_tokens"),
                "estimated_spend": number(spend.get("total")),
                "http_attempts": metrics.get("http_attempts"),
                "model_calls": metrics.get("model_calls"),
                "failure_categories": sorted(failures | failed),
                "critical_categories": sorted(failed & CRITICAL_CATEGORIES),
            }
            trials_by_case.setdefault(result["case_id"], []).append(trial)
        environment = report["environment"]
        exported.append(
            {
                "run_id": identifier(report["run_id"]),
                "status": report["status"],
                "model": identifier(report["model"]),
                "runtime": identifier(report["runtime"]),
                "suite": identifier(report["suite"]["name"]),
                "suite_fingerprint": report["suite"]["fingerprint"],
                "configuration_fingerprint": report["configuration_fingerprint"],
                "repeat_count": report["configuration"]["repeat_count"],
                "shuffle_seed": report["configuration"].get("shuffle_seed"),
                "git_revision": identifier(environment.get("git_revision")),
                "git_dirty": environment.get("git_dirty"),
                "python": identifier(environment.get("python")),
                "platform": identifier(
                    str(environment.get("platform", "")).split("-")[0]
                ),
                "cases": {
                    case: {
                        **summarize_trials(trials),
                        "trial_results": sorted(
                            trials, key=lambda item: item["trial_index"]
                        ),
                    }
                    for case, trials in sorted(trials_by_case.items())
                },
            }
        )
    return {"kind": EVIDENCE_KIND, "schema_version": 1, "reports": exported}

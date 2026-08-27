from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

from .store import LocalTraceStore


def compare_runs(store: LocalTraceStore, base_run_id: str, head_run_id: str) -> dict[str, Any]:
    base_run = _run_metadata(store, base_run_id)
    head_run = _run_metadata(store, head_run_id)
    base_cases = _case_trials_by_id(store, base_run_id)
    head_cases = _case_trials_by_id(store, head_run_id)
    compatible, compatibility_warnings = _compatibility(base_run, head_run)
    case_ids = sorted(set(base_cases) | set(head_cases))
    comparisons: list[dict[str, Any]] = []
    for case_id in case_ids:
        base_trials = base_cases.get(case_id)
        head_trials = head_cases.get(case_id)
        if base_trials is None:
            comparisons.append(
                {
                    "case_id": case_id,
                    "category": "new_case" if compatible else "incompatible",
                    "details": {"head": _aggregate_case(head_trials or [])},
                }
            )
            continue
        if head_trials is None:
            comparisons.append(
                {
                    "case_id": case_id,
                    "category": "missing_case" if compatible else "incompatible",
                    "details": {"base": _aggregate_case(base_trials)},
                }
            )
            continue
        base = _aggregate_case(base_trials)
        head = _aggregate_case(head_trials)
        category = (
            _status_category(base["pass_rate"], head["pass_rate"])
            if compatible
            else "incompatible"
        )
        details = _case_delta(base, head)
        comparison = {"case_id": case_id, "category": category, "details": details}
        regression = _regression_category(details) if compatible else None
        if regression is not None and category == "unchanged_pass":
            comparison["category"] = regression
        comparisons.append(comparison)
    return {
        "base_run_id": base_run_id,
        "head_run_id": head_run_id,
        "base_run": _run_overview(base_run),
        "head_run": _run_overview(head_run),
        "compatible": compatible,
        "compatibility_warnings": compatibility_warnings,
        "summary": _summary(comparisons),
        "cases": comparisons,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m mikucli.observability.compare")
    parser.add_argument("--base", required=True, help="Base benchmark run ID.")
    parser.add_argument("--head", required=True, help="Head benchmark run ID.")
    parser.add_argument(
        "--store-root",
        default=str(Path.cwd() / ".mikucli" / "observability"),
        help="Observability store root. Defaults to .mikucli/observability in the current directory.",
    )
    args = parser.parse_args(argv)
    store = LocalTraceStore(Path(args.store_root), mode="sqlite")
    print(json.dumps(compare_runs(store, args.base, args.head), indent=2))
    return 0


def _case_trials_by_id(store: LocalTraceStore, run_id: str) -> dict[str, list[dict[str, Any]]]:
    rows = store.fetch_all("select * from eval_cases where run_id = ?", (run_id,))
    cases: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        row["passed"] = bool(row["passed"])
        row["metrics"] = _loads(row.pop("metrics_json"))
        row["failure_reasons"] = _loads(row.pop("failure_reasons_json"))
        row["changed_paths"] = _loads(row.pop("changed_paths_json"))
        cases.setdefault(str(row["case_id"]), []).append(row)
    for trials in cases.values():
        trials.sort(key=lambda row: int(row.get("trial_index") or 1))
    return cases


def _run_metadata(store: LocalTraceStore, run_id: str) -> dict[str, Any]:
    row = store.fetch_one("select * from runs where run_id = ?", (run_id,))
    if row is None:
        return {}
    summary = _loads(row.get("summary_json") or "")
    row["summary"] = summary if isinstance(summary, dict) else {}
    return row


def _run_overview(run: dict[str, Any]) -> dict[str, Any]:
    summary = run.get("summary") if isinstance(run.get("summary"), dict) else {}
    return {
        "run_id": run.get("run_id"),
        "runtime": summary.get("runtime") or "unknown",
        "model": run.get("model") or "unknown",
        "started_at": run.get("started_at") or "unknown",
        "passed_cases": run.get("passed_cases") or 0,
        "total_cases": run.get("total_cases") or 0,
        "success_rate": run.get("success_rate") or 0.0,
    }


def _compatibility(base: dict[str, Any], head: dict[str, Any]) -> tuple[bool, list[str]]:
    warnings: list[str] = []
    compatible = True
    base_summary = base.get("summary") if isinstance(base.get("summary"), dict) else {}
    head_summary = head.get("summary") if isinstance(head.get("summary"), dict) else {}
    for label, key in (
        ("suite fingerprint", "suite_fingerprint"),
        ("configuration fingerprint", "configuration_fingerprint"),
    ):
        base_value = base_summary.get(key)
        head_value = head_summary.get(key)
        if base_value is None or head_value is None:
            warnings.append(f"{label} is missing from one or both runs")
            compatible = False
        elif base_value != head_value:
            warnings.append(f"{label} differs")
            # Configuration is an experiment dimension (for example native
            # versus LangGraph runtime), so keep the comparison available and
            # disclose the difference. A suite mismatch still makes case
            # deltas unsafe to classify as regressions.
            if key == "suite_fingerprint":
                compatible = False
    return compatible, warnings


def _status_category(base_pass_rate: float, head_pass_rate: float) -> str:
    if 0 < head_pass_rate < 1:
        return "flaky_failure"
    if base_pass_rate == 1 and head_pass_rate == 0:
        return "new_failure"
    if base_pass_rate < 1 and head_pass_rate == 1:
        return "recovered"
    if base_pass_rate == 1 and head_pass_rate == 1:
        return "unchanged_pass"
    return "unchanged_fail"


def _case_delta(base: dict[str, Any], head: dict[str, Any]) -> dict[str, Any]:
    base_metrics = base["metrics"]
    head_metrics = head["metrics"]
    base_reasons = _failure_reason_set(base)
    head_reasons = _failure_reason_set(head)
    return {
        "base_passed": base["pass_rate"] == 1,
        "head_passed": head["pass_rate"] == 1,
        "base_pass_rate": base["pass_rate"],
        "head_pass_rate": head["pass_rate"],
        "base_trial_count": base["trial_count"],
        "head_trial_count": head["trial_count"],
        "new_failure_reasons": sorted(head_reasons - base_reasons),
        "resolved_failure_reasons": sorted(base_reasons - head_reasons),
        "tool_call_count_delta": _delta(
            _metric(head_metrics, "tool_call_count"),
            _metric(base_metrics, "tool_call_count"),
        ),
        "model_retry_delta": _delta(
            _metric(head_metrics, "model_retries"),
            _metric(base_metrics, "model_retries"),
        ),
        "step_retry_delta": _delta(
            _metric(head_metrics, "step_retries"),
            _metric(base_metrics, "step_retries"),
        ),
        "elapsed_seconds_delta": _delta(
            _metric(head_metrics, "elapsed_seconds"),
            _metric(base_metrics, "elapsed_seconds"),
        ),
        "total_token_delta": _delta(
            _cost_metric(head_metrics, "total_tokens"),
            _cost_metric(base_metrics, "total_tokens"),
        ),
        "base_metrics": base_metrics,
        "head_metrics": head_metrics,
    }


def _regression_category(details: dict[str, Any]) -> str | None:
    base_elapsed = _metric(details["base_metrics"], "elapsed_seconds")
    head_elapsed = _metric(details["head_metrics"], "elapsed_seconds")
    base_tokens = _cost_metric(details["base_metrics"], "total_tokens")
    head_tokens = _cost_metric(details["head_metrics"], "total_tokens")
    if (
        base_elapsed is not None
        and head_elapsed is not None
        and head_elapsed >= base_elapsed * 1.25
        and head_elapsed - base_elapsed >= 5
    ):
        return "latency_regression"
    if (
        base_tokens is not None
        and head_tokens is not None
        and head_tokens >= base_tokens * 1.25
        and head_tokens - base_tokens >= 1000
    ):
        return "cost_regression"
    if (
        _increased_from_zero(
            _metric(details["base_metrics"], "model_retries"),
            _metric(details["head_metrics"], "model_retries"),
        )
        or _increased_from_zero(
            _metric(details["base_metrics"], "step_retries"),
            _metric(details["head_metrics"], "step_retries"),
        )
    ):
        return "tool_retry_regression"
    return None


def _summary(comparisons: list[dict[str, Any]]) -> dict[str, Any]:
    categories: dict[str, int] = {}
    for comparison in comparisons:
        category = str(comparison["category"])
        categories[category] = categories.get(category, 0) + 1
    return {"total_cases": len(comparisons), "categories": categories}


def _failure_reason_set(case: dict[str, Any]) -> set[str]:
    reasons = case.get("failure_reasons") or []
    values = []
    for reason in reasons:
        if not isinstance(reason, dict):
            continue
        values.append(f"{reason.get('category')}:{reason.get('source')}:{reason.get('message')}")
    return set(values)


def _metric(metrics: dict[str, Any], name: str) -> float | None:
    value = metrics.get(name)
    return float(value) if isinstance(value, (int, float)) else None


def _cost_metric(metrics: dict[str, Any], name: str) -> float | None:
    cost = metrics.get("cost")
    if not isinstance(cost, dict):
        return None
    value = cost.get(name)
    return float(value) if isinstance(value, (int, float)) else None


def _delta(head: float | None, base: float | None) -> float | None:
    if head is None or base is None:
        return None
    return head - base


def _aggregate_case(trials: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = [trial.get("metrics") if isinstance(trial.get("metrics"), dict) else {} for trial in trials]
    return {
        "pass_rate": sum(bool(trial.get("passed")) for trial in trials) / len(trials),
        "trial_count": len(trials),
        "failure_reasons": [
            reason
            for trial in trials
            for reason in (trial.get("failure_reasons") or [])
            if isinstance(reason, dict)
        ],
        "metrics": {
            "tool_call_count": _median_metric(metrics, "tool_call_count"),
            "model_retries": _median_metric(metrics, "model_retries"),
            "step_retries": _median_metric(metrics, "step_retries"),
            "elapsed_seconds": _median_metric(metrics, "elapsed_seconds"),
            "cost": {
                "total_tokens": _median_cost_metric(metrics, "total_tokens"),
            },
        },
    }


def _median_metric(metrics: list[dict[str, Any]], name: str) -> float | None:
    values = [_metric(value, name) for value in metrics]
    if not values or any(value is None for value in values):
        return None
    return float(statistics.median(value for value in values if value is not None))


def _median_cost_metric(metrics: list[dict[str, Any]], name: str) -> float | None:
    values = [_cost_metric(value, name) for value in metrics]
    if not values or any(value is None for value in values):
        return None
    return float(statistics.median(value for value in values if value is not None))


def _increased_from_zero(base: float | None, head: float | None) -> bool:
    return base == 0 and head is not None and head > 0


def _loads(raw: str) -> Any:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


if __name__ == "__main__":
    raise SystemExit(main())

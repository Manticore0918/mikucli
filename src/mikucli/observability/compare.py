from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .store import LocalTraceStore


def compare_runs(store: LocalTraceStore, base_run_id: str, head_run_id: str) -> dict[str, Any]:
    base_cases = _cases_by_id(store, base_run_id)
    head_cases = _cases_by_id(store, head_run_id)
    case_ids = sorted(set(base_cases) | set(head_cases))
    comparisons: list[dict[str, Any]] = []
    for case_id in case_ids:
        base = base_cases.get(case_id)
        head = head_cases.get(case_id)
        if base is None:
            comparisons.append({"case_id": case_id, "category": "new_case", "details": {"head": head}})
            continue
        if head is None:
            comparisons.append({"case_id": case_id, "category": "missing_case", "details": {"base": base}})
            continue
        category = _status_category(bool(base["passed"]), bool(head["passed"]))
        details = _case_delta(base, head)
        comparison = {"case_id": case_id, "category": category, "details": details}
        regression = _regression_category(details)
        if regression is not None and category == "unchanged_pass":
            comparison["category"] = regression
        comparisons.append(comparison)
    return {
        "base_run_id": base_run_id,
        "head_run_id": head_run_id,
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


def _cases_by_id(store: LocalTraceStore, run_id: str) -> dict[str, dict[str, Any]]:
    rows = store.fetch_all("select * from eval_cases where run_id = ?", (run_id,))
    cases: dict[str, dict[str, Any]] = {}
    for row in rows:
        row["passed"] = bool(row["passed"])
        row["metrics"] = _loads(row.pop("metrics_json"))
        row["failure_reasons"] = _loads(row.pop("failure_reasons_json"))
        row["changed_paths"] = _loads(row.pop("changed_paths_json"))
        cases[str(row["case_id"])] = row
    return cases


def _status_category(base_passed: bool, head_passed: bool) -> str:
    if base_passed and not head_passed:
        return "new_failure"
    if not base_passed and head_passed:
        return "recovered"
    if base_passed and head_passed:
        return "unchanged_pass"
    return "unchanged_fail"


def _case_delta(base: dict[str, Any], head: dict[str, Any]) -> dict[str, Any]:
    base_metrics = base["metrics"]
    head_metrics = head["metrics"]
    base_reasons = _failure_reason_set(base)
    head_reasons = _failure_reason_set(head)
    return {
        "base_passed": base["passed"],
        "head_passed": head["passed"],
        "new_failure_reasons": sorted(head_reasons - base_reasons),
        "resolved_failure_reasons": sorted(base_reasons - head_reasons),
        "tool_call_count_delta": _metric(head_metrics, "tool_call_count") - _metric(base_metrics, "tool_call_count"),
        "model_retry_delta": _metric(head_metrics, "model_retries") - _metric(base_metrics, "model_retries"),
        "step_retry_delta": _metric(head_metrics, "step_retries") - _metric(base_metrics, "step_retries"),
        "elapsed_seconds_delta": _metric(head_metrics, "elapsed_seconds") - _metric(base_metrics, "elapsed_seconds"),
        "llm_latency_seconds_delta": _metric(head_metrics, "llm_latency_seconds") - _metric(base_metrics, "llm_latency_seconds"),
        "agent_latency_seconds_delta": _metric(head_metrics, "agent_latency_seconds") - _metric(base_metrics, "agent_latency_seconds"),
        "total_token_delta": _cost_metric(head_metrics, "total_tokens") - _cost_metric(base_metrics, "total_tokens"),
        "base_metrics": base_metrics,
        "head_metrics": head_metrics,
    }


def _regression_category(details: dict[str, Any]) -> str | None:
    base_elapsed = _metric(details["base_metrics"], "elapsed_seconds")
    head_elapsed = _metric(details["head_metrics"], "elapsed_seconds")
    base_tokens = _cost_metric(details["base_metrics"], "total_tokens")
    head_tokens = _cost_metric(details["head_metrics"], "total_tokens")
    if head_elapsed >= base_elapsed * 1.25 and head_elapsed - base_elapsed >= 5:
        return "latency_regression"
    if head_tokens >= base_tokens * 1.25 and head_tokens - base_tokens >= 1000:
        return "cost_regression"
    if (
        details["model_retry_delta"] > 0
        or details["step_retry_delta"] > 0
        or (_metric(details["base_metrics"], "model_retries") == 0 and _metric(details["head_metrics"], "model_retries") > 0)
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


def _metric(metrics: dict[str, Any], name: str) -> float:
    value = metrics.get(name)
    return float(value) if isinstance(value, (int, float)) else 0.0


def _cost_metric(metrics: dict[str, Any], name: str) -> float:
    cost = metrics.get("cost")
    if not isinstance(cost, dict):
        return 0.0
    value = cost.get(name)
    return float(value) if isinstance(value, (int, float)) else 0.0


def _loads(raw: str) -> Any:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


if __name__ == "__main__":
    raise SystemExit(main())

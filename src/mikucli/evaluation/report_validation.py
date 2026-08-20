from __future__ import annotations

import math
from typing import Any

REPORT_SCHEMA_VERSION = 2
_REPORT_STATUSES = {"running", "stopped", "interrupted", "complete"}
_SESSION_MODES = {
    "built_in_single_agent",
    "built_in_multi_agent",
    "mcp_single_agent",
    "mcp_multi_agent",
}


def validate_report_payload(payload: Any, *, allow_legacy: bool = False) -> dict[str, Any]:
    """Validate report structure and invariants before resume or database import."""
    report = _mapping(payload, "report")
    version = report.get("schema_version")
    if version is None and allow_legacy:
        _validate_results(report, repeat_count=None, case_ids=None)
        return report
    if version != REPORT_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported benchmark report schema version {version!r}; "
            f"expected {REPORT_SCHEMA_VERSION}"
        )

    run_id = _nonempty_string(report.get("run_id"), "run_id")
    run_group_id = _nonempty_string(report.get("run_group_id"), "run_group_id")
    if run_group_id != run_id:
        raise ValueError("run_group_id must match run_id")
    status = _nonempty_string(report.get("status"), "status")
    if status not in _REPORT_STATUSES:
        raise ValueError(f"status must be one of {sorted(_REPORT_STATUSES)}")
    suite = _mapping(report.get("suite"), "suite")
    _fingerprint(suite.get("fingerprint"), "suite.fingerprint")
    case_ids = _string_list(suite.get("case_ids"), "suite.case_ids")
    if not case_ids or len(case_ids) != len(set(case_ids)):
        raise ValueError("suite.case_ids must contain unique, non-empty case IDs")
    configuration = _mapping(report.get("configuration"), "configuration")
    repeat_count = _positive_int(configuration.get("repeat_count"), "configuration.repeat_count")
    _fingerprint(report.get("configuration_fingerprint"), "configuration_fingerprint")
    _mapping(report.get("environment"), "environment")

    results = _validate_results(report, repeat_count=repeat_count, case_ids=set(case_ids))
    for index, result in enumerate(results):
        result_group = result.get("run_group_id")
        if result_group not in {None, "", run_id}:
            raise ValueError(f"results[{index}].run_group_id does not match run_id")

    summary = _mapping(report.get("summary"), "summary")
    completed = _nonnegative_int(summary.get("completed_trials"), "summary.completed_trials")
    planned = _nonnegative_int(summary.get("planned_trials"), "summary.planned_trials")
    incomplete = _nonnegative_int(summary.get("incomplete_trials"), "summary.incomplete_trials")
    total_cases = _nonnegative_int(summary.get("total_cases"), "summary.total_cases")
    passed_cases = _nonnegative_int(summary.get("passed_cases"), "summary.passed_cases")
    success_rate = _nonnegative_number(summary.get("success_rate"), "summary.success_rate")
    expected_planned = len(case_ids) * repeat_count
    if planned != expected_planned:
        raise ValueError("summary.planned_trials does not match selected cases and repeat count")
    if completed != len(results) or total_cases != len(results):
        raise ValueError("summary completed counts do not match results")
    if incomplete != planned - completed or completed > planned:
        raise ValueError("summary incomplete trial count is inconsistent")
    if passed_cases != sum(result["passed"] is True for result in results):
        raise ValueError("summary.passed_cases does not match results")
    expected_rate = round(passed_cases / completed, 4) if completed else 0.0
    if success_rate > 1 or not math.isclose(success_rate, expected_rate, abs_tol=0.00005):
        raise ValueError("summary.success_rate does not match results")
    if status == "complete" and completed != planned:
        raise ValueError("a complete report must contain every planned trial")
    return report


def _validate_results(
    report: dict[str, Any],
    *,
    repeat_count: int | None,
    case_ids: set[str] | None,
) -> list[dict[str, Any]]:
    raw_results = report.get("results")
    if not isinstance(raw_results, list):
        raise ValueError("results must be a list")
    results: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()
    for index, raw in enumerate(raw_results):
        result = _mapping(raw, f"results[{index}]")
        case_id = _nonempty_string(result.get("case_id"), f"results[{index}].case_id")
        if case_ids is not None and case_id not in case_ids:
            raise ValueError(f"results[{index}].case_id is not in suite.case_ids")
        if not isinstance(result.get("passed"), bool):
            raise ValueError(f"results[{index}].passed must be a boolean")
        trial_index = _positive_int(result.get("trial_index", 1), f"results[{index}].trial_index")
        trial_count = _positive_int(result.get("trial_count", 1), f"results[{index}].trial_count")
        if trial_index > trial_count:
            raise ValueError(f"results[{index}].trial_index exceeds trial_count")
        if repeat_count is not None and trial_count != repeat_count:
            raise ValueError(f"results[{index}].trial_count does not match configuration")
        key = (case_id, trial_index)
        if key in seen:
            raise ValueError(f"duplicate benchmark result for {case_id} trial {trial_index}")
        seen.add(key)
        mode = result.get("session_mode")
        if repeat_count is not None and mode not in _SESSION_MODES:
            raise ValueError(f"results[{index}].session_mode is invalid")
        if repeat_count is not None:
            _nonempty_string(result.get("task_id"), f"results[{index}].task_id")
            _nonempty_string(result.get("model"), f"results[{index}].model")
            _nonempty_string(result.get("workspace"), f"results[{index}].workspace")
            for name in (
                "check_results",
                "changed_paths",
                "tool_calls",
                "approvals",
                "failure_reasons",
                "tool_correctness_results",
            ):
                if not isinstance(result.get(name), list):
                    raise ValueError(f"results[{index}].{name} must be a list")
        metrics = _mapping(result.get("metrics"), f"results[{index}].metrics")
        for name in ("tool_call_count", "model_retries", "step_retries"):
            _nonnegative_int(metrics.get(name, 0), f"results[{index}].metrics.{name}")
        _nonnegative_number(
            metrics.get("elapsed_seconds", 0.0),
            f"results[{index}].metrics.elapsed_seconds",
        )
        cost = metrics.get("cost")
        if cost is not None:
            for name, value in _mapping(cost, f"results[{index}].metrics.cost").items():
                if value is not None:
                    _nonnegative_int(value, f"results[{index}].metrics.cost.{name}")
        results.append(result)
    return results


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JSON object")
    return value


def _nonempty_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _string_list(value: Any, name: str) -> list[str]:
    if not isinstance(value, list) or any(not isinstance(item, str) or not item for item in value):
        raise ValueError(f"{name} must be a list of non-empty strings")
    return value


def _positive_int(value: Any, name: str) -> int:
    result = _nonnegative_int(value, name)
    if result == 0:
        raise ValueError(f"{name} must be positive")
    return result


def _nonnegative_int(value: Any, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _nonnegative_number(value: Any, name: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{name} must be a non-negative finite number")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"{name} must be a non-negative finite number")
    return result


def _fingerprint(value: Any, name: str) -> str:
    result = _nonempty_string(value, name)
    if len(result) != 64 or any(character not in "0123456789abcdef" for character in result):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return result

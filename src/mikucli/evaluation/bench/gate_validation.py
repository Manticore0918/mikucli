from __future__ import annotations

import re
from collections import Counter
from typing import Any

from ..report_validation import validate_report_payload
from .evidence import CRITICAL_CATEGORIES, EVIDENCE_KIND, number, summarize_trials
from .gate_policy import catalog
from .metadata import _fingerprint, suite_fingerprint

DIGEST = re.compile(r"^[a-f0-9]{64}$")


def validate_reports(reports: list[dict[str, Any]]) -> None:
    known = catalog()
    for report in reports:
        validate_report_payload(report)
        if report["configuration_fingerprint"] != _fingerprint(report["configuration"]):
            raise ValueError(
                "report configuration fingerprint does not match its contents"
            )
        names = report["suite"]["name"].split("+")
        if report["suite"]["fingerprint"] != suite_fingerprint(names):
            raise ValueError(
                "report was produced by a different fixture/scorer version"
            )
        if report.get("model") != report["configuration"].get("model"):
            raise ValueError("report model does not match configuration")
        if report.get("runtime") != report["configuration"].get("single_agent_runtime"):
            raise ValueError("report runtime does not match configuration")
        for raw in report["results"]:
            case = known.get(raw["case_id"])
            if case is None or case.task.suite not in names:
                raise ValueError("report contains a case outside its declared suite")
            if (raw["task_id"], raw["session_mode"], raw["model"]) != (
                case.task.id,
                case.session_mode.value,
                report["model"],
            ):
                raise ValueError(
                    "result identity does not match the catalog/configuration"
                )
            checks = raw["check_results"] + raw["tool_correctness_results"]
            if not checks or any(
                not isinstance(check, dict) or type(check.get("passed")) is not bool
                for check in checks
            ):
                raise ValueError("result lacks valid scorer evidence")
            if not any(
                check.get("name") == "task_outcome" for check in raw["check_results"]
            ):
                raise ValueError("result lacks explicit runtime-outcome scoring")
            expected_checks = Counter(
                [
                    "task_outcome",
                    *(
                        getattr(check, "__name__", "check")
                        for check in case.task.checks
                    ),
                ]
            )
            if (
                Counter(check.get("name") for check in raw["check_results"])
                != expected_checks
            ):
                raise ValueError(
                    "result is missing declared task checks or includes duplicate scorer evidence"
                )
            if any(
                check.get("category")
                not in CRITICAL_CATEGORIES | {"task_success", "heldout_behavior"}
                for check in checks
            ):
                raise ValueError("unknown scorer category")
            outcome_matches = raw.get(
                "outcome_status"
            ) == case.task.expected_status and (
                case.task.expected_reason is None
                or raw.get("outcome_reason") == case.task.expected_reason
            )
            if raw["passed"] and (
                not outcome_matches or any(not check["passed"] for check in checks)
            ):
                raise ValueError("passed result contradicts runtime/scorer evidence")
            if any(
                check["passed"] != outcome_matches
                for check in raw["check_results"]
                if check.get("name") == "task_outcome"
            ):
                raise ValueError(
                    "runtime-outcome check contradicts the catalog contract"
                )
            spend = raw["metrics"].get("estimated_spend")
            if spend is not None and (
                not isinstance(spend, dict)
                or any(
                    value is not None and number(value) is None
                    for value in spend.values()
                )
            ):
                raise ValueError("result contains invalid estimated spend")


def validate_evidence(payload: dict[str, Any]) -> dict[str, Any]:
    if (
        payload.get("kind") != EVIDENCE_KIND
        or type(payload.get("schema_version")) is not int
        or payload["schema_version"] != 1
    ):
        raise ValueError("unsupported evidence artifact")
    reports = payload.get("reports")
    if not isinstance(reports, list) or not reports:
        raise ValueError("evidence requires reports")
    known = catalog()
    seen = set()
    for report in reports:
        if not isinstance(report, dict):
            raise ValueError("invalid evidence report")
        for name in ("suite_fingerprint", "configuration_fingerprint"):
            if not isinstance(report.get(name), str) or not DIGEST.fullmatch(
                report[name]
            ):
                raise ValueError("invalid evidence fingerprint")
        repeat = report.get("repeat_count")
        if type(repeat) is not int or repeat < 1:
            raise ValueError("invalid evidence repeat count")
        cases = report.get("cases")
        if not isinstance(cases, dict) or not cases:
            raise ValueError("evidence requires per-case trials")
        for case_id, case in cases.items():
            if case_id not in known or case_id in seen or not isinstance(case, dict):
                raise ValueError("unknown or duplicated evidence case")
            seen.add(case_id)
            trials = case.get("trial_results")
            if not isinstance(trials, list) or not trials:
                raise ValueError("evidence case requires trials")
            indices = []
            for trial in trials:
                if not isinstance(trial, dict) or type(trial.get("passed")) is not bool:
                    raise ValueError("invalid evidence trial")
                index = trial.get("trial_index")
                if (
                    type(index) is not int
                    or not 1 <= index <= repeat
                    or index in indices
                ):
                    raise ValueError("invalid or duplicate evidence trial index")
                indices.append(index)
                if number(trial.get("elapsed_seconds")) is None:
                    raise ValueError("invalid trial latency")
                for metric in ("prompt_tokens", "completion_tokens", "total_tokens"):
                    value = trial.get(metric)
                    if value is not None and (type(value) is not int or value < 0):
                        raise ValueError("invalid trial token usage")
                if (
                    trial.get("estimated_spend") is not None
                    and number(trial["estimated_spend"]) is None
                ):
                    raise ValueError("invalid trial spend")
                for field in ("failure_categories", "critical_categories"):
                    if not isinstance(trial.get(field), list) or any(
                        not isinstance(value, str) for value in trial[field]
                    ):
                        raise ValueError("invalid trial categories")
                if set(trial["critical_categories"]) - CRITICAL_CATEGORIES:
                    raise ValueError("unknown critical category")
                if trial["passed"] and trial["critical_categories"]:
                    raise ValueError("passed trial contains critical failures")
                contract = known[case_id].task
                if trial["passed"] and (
                    trial.get("outcome_status") != contract.expected_status
                    or (
                        contract.expected_reason is not None
                        and trial.get("outcome_reason") != contract.expected_reason
                    )
                ):
                    raise ValueError("passed evidence contradicts the catalog outcome")
                for field in ("http_attempts", "model_calls"):
                    value = trial.get(field)
                    if value is not None and (type(value) is not int or value < 0):
                        raise ValueError("invalid HTTP/model attempt metric")
            if report.get("status") == "complete" and set(indices) != set(
                range(1, repeat + 1)
            ):
                raise ValueError("complete evidence is missing planned trials")
            case.update(
                summarize_trials(trials)
            )  # Recompute; never trust supplied aggregates.
    return payload

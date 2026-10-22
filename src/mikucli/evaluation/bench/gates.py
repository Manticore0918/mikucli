"""Release decisions over complete, comparable per-case trial evidence."""

from __future__ import annotations

import re
from typing import Any

from .gate_policy import validate_policy
from .gate_validation import validate_evidence

RATIOS = {
    "median_latency_seconds": "maximum_median_latency_ratio",
    "p95_latency_seconds": "maximum_p95_latency_ratio",
    "median_tokens": "maximum_median_tokens_ratio",
    "median_spend": "maximum_median_spend_ratio",
}
COMPATIBILITY = (
    "suite_fingerprint",
    "configuration_fingerprint",
    "model",
    "runtime",
    "repeat_count",
    "shuffle_seed",
    "platform",
    "python",
)


def evaluate_gate(
    policy: dict[str, Any],
    evidence: dict[str, Any],
    baseline: dict[str, Any] | None = None,
) -> dict[str, Any]:
    validate_policy(policy)
    validate_evidence(evidence)
    if baseline is not None:
        validate_evidence(baseline)
    failures: list[str] = []
    indexed = _index(evidence, policy, failures)
    baseline_index = (
        _index(baseline, policy, failures, prefix="baseline ")
        if baseline is not None
        else {}
    )
    if policy["require_baseline"] and baseline is None:
        failures.append("a reviewed, compatible baseline is required")
    missing = set(policy["cases"]) - set(indexed)
    extra = set(indexed) - set(policy["cases"])
    failures.extend(f"missing case: {case}" for case in sorted(missing))
    failures.extend(f"unexpected case: {case}" for case in sorted(extra))
    comparisons = {}
    for case_id, rule in policy["cases"].items():
        if case_id not in indexed:
            continue
        report, case = indexed[case_id]
        if case["critical_failures"]:
            failures.append(f"{case_id}: critical contract/safety/scope failure")
        if case["pass_rate"] < rule["minimum_pass_rate"]:
            failures.append(f"{case_id}: pass rate below policy")
        if case["ci95_low"] < rule.get("minimum_ci95_low", 0):
            failures.append(f"{case_id}: insufficient confidence for policy")
        if policy["require_usage"] and any(
            trial[field] is None
            for trial in case["trial_results"]
            for field in (
                "prompt_tokens",
                "completion_tokens",
                "total_tokens",
                "estimated_spend",
            )
        ):
            failures.append(f"{case_id}: complete usage and spend evidence is required")
        for metric in RATIOS:
            ceiling = rule.get("maximum_" + metric)
            if ceiling is not None and (case[metric] is None or case[metric] > ceiling):
                failures.append(f"{case_id}: {metric} missing or above policy")
        if baseline is None:
            continue
        if case_id not in baseline_index:
            failures.append(f"{case_id}: baseline is missing this case")
            continue
        base_report, base_case = baseline_index[case_id]
        if any(report.get(key) != base_report.get(key) for key in COMPATIBILITY):
            failures.append(
                f"{case_id}: incompatible baseline configuration, harness, or environment"
            )
            continue
        if (
            base_case["critical_failures"]
            or base_case["pass_rate"] < rule["minimum_pass_rate"]
        ):
            failures.append(f"{case_id}: baseline does not meet the quality policy")
        if (
            base_case["pass_rate"] - case["pass_rate"]
            > policy.get("maximum_pass_rate_drop", 0) + 1e-12
        ):
            failures.append(f"{case_id}: pass rate regressed")
        base_trials = {
            trial["trial_index"]: trial for trial in base_case["trial_results"]
        }
        if set(base_trials) != {
            trial["trial_index"] for trial in case["trial_results"]
        }:
            failures.append(f"{case_id}: baseline trial schedule is incomplete")
            continue
        comparisons[case_id] = {
            "paired_losses": sum(
                base_trials[trial["trial_index"]]["passed"] and not trial["passed"]
                for trial in case["trial_results"]
            ),
            "paired_wins": sum(
                not base_trials[trial["trial_index"]]["passed"] and trial["passed"]
                for trial in case["trial_results"]
            ),
        }
        for metric, setting in RATIOS.items():
            if setting not in policy:
                continue
            head, base = case[metric], base_case[metric]
            if head is None or base is None:
                failures.append(f"{case_id}: {metric} unavailable for regression gate")
            elif head > base * policy[setting]:
                failures.append(f"{case_id}: {metric} regressed beyond policy")
    return {
        "passed": not failures,
        "policy": policy["name"],
        "failures": failures,
        "comparisons": comparisons,
    }


def _index(evidence, policy, failures, prefix=""):
    indexed = {}
    for report in evidence["reports"]:
        if report["status"] != "complete":
            failures.append(prefix + "report is incomplete")
        if report["repeat_count"] < policy["minimum_repeat_count"]:
            failures.append(prefix + "report has too few repeated trials")
        if policy["require_clean_revision"] and (
            report.get("git_dirty") is not False
            or re.fullmatch(
                r"(?:[a-f0-9]{40}|[a-f0-9]{64})", report.get("git_revision", "")
            )
            is None
        ):
            failures.append(
                prefix + "report requires an identified, clean Git revision"
            )
        for case_id, case in report["cases"].items():
            if case["trials"] != report["repeat_count"]:
                failures.append(prefix + f"{case_id}: missing planned trials")
            indexed[case_id] = report, case
    return indexed

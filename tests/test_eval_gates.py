from __future__ import annotations

import copy
import json

import pytest

from mikucli.evaluation.bench.evidence import export_evidence
from mikucli.evaluation.bench.gate_cli import main
from mikucli.evaluation.bench.gate_policy import (
    load_policy,
    strict_json,
    validate_policy,
)
from mikucli.evaluation.bench.gate_validation import validate_reports
from mikucli.evaluation.bench.gates import evaluate_gate
from mikucli.evaluation.bench.persistence import load_report
from mikucli.evaluation.bench.reliability import ReliabilityRunner
from mikucli.evaluation.bench.suites import benchmark_cases

CASE = "provider_retry_recovery:built_in_single_agent"


@pytest.fixture
def report(tmp_path):
    case = next(case for case in benchmark_cases("reliability") if case.id == CASE)
    _, path, _ = ReliabilityRunner(
        root=tmp_path, repeat_count=3, shuffle_seed=1729
    ).run([case])
    return load_report(path), path


@pytest.fixture
def policy():
    return {
        "schema_version": 1,
        "name": "test-policy",
        "minimum_repeat_count": 3,
        "require_baseline": False,
        "require_usage": False,
        "require_clean_revision": False,
        "maximum_pass_rate_drop": 0,
        "cases": {CASE: {"minimum_pass_rate": 1.0}},
    }


def artifact(report):
    raw, _ = report
    return export_evidence([raw])


def trials(evidence):
    return evidence["reports"][0]["cases"][CASE]["trial_results"]


def test_complete_gate_and_cli_emit_only_sanitized_evidence(report, policy, tmp_path):
    raw, path = report
    raw["results"][0]["final_answer"] = "SECRET-ANSWER"
    raw["results"][0]["tool_calls"] = [
        {"arguments": {"password": "SECRET-ARGUMENT"}, "content": "SECRET-TOOL"}
    ]
    raw["environment"]["executable"] = "SECRET-LOCAL-PATH"
    path.write_text(json.dumps(raw), encoding="utf-8")
    policy_path = tmp_path / "policy.json"
    policy_path.write_text(json.dumps(policy), encoding="utf-8")
    output = tmp_path / "evidence.json"
    assert (
        main(
            [
                "--policy",
                str(policy_path),
                "--report",
                str(path),
                "--output",
                str(output),
            ]
        )
        == 0
    )
    text = output.read_text(encoding="utf-8")
    assert "SECRET-" not in text
    evidence = json.loads(text)
    stats = evidence["reports"][0]["cases"][CASE]
    assert stats["trials"] == stats["passed"] == 3
    assert 0 < stats["ci95_low"] < 1
    assert not stats["flaky"]
    assert raw["case_summary"][CASE]["trials"] == 3


@pytest.mark.parametrize(
    "mutation", ["missing", "incomplete", "few_repeats", "unclean", "missing_baseline"]
)
def test_gate_fails_closed_for_missing_evidence(report, policy, mutation):
    evidence = artifact(report)
    if mutation == "missing":
        policy["cases"]["provider_exhaustion:built_in_single_agent"] = {
            "minimum_pass_rate": 1
        }
    elif mutation == "incomplete":
        evidence["reports"][0]["status"] = "stopped"
        trials(evidence).pop()
    elif mutation == "few_repeats":
        policy["minimum_repeat_count"] = 4
    elif mutation == "unclean":
        policy["require_clean_revision"] = True
    else:
        policy["require_baseline"] = True
    assert not evaluate_gate(policy, evidence)["passed"]


def test_safety_failures_cannot_be_averaged_away(report, policy):
    evidence = artifact(report)
    policy["cases"][CASE]["minimum_pass_rate"] = 0
    trials(evidence)[0].update(
        passed=False, critical_categories=["safety"], failure_categories=["safety"]
    )
    result = evaluate_gate(policy, evidence)
    assert not result["passed"]
    assert any("critical" in message for message in result["failures"])


def test_matching_baseline_and_paired_regression(report, policy):
    baseline = artifact(report)
    head = copy.deepcopy(baseline)
    assert evaluate_gate(policy, head, baseline)["passed"]
    trials(head)[0]["passed"] = False
    policy["cases"][CASE]["minimum_pass_rate"] = 0.5
    result = evaluate_gate(policy, head, baseline)
    assert not result["passed"]
    assert result["comparisons"][CASE]["paired_losses"] == 1
    assert head["reports"][0]["cases"][CASE]["flaky"]


def test_latency_token_and_spend_regressions_are_gated(report, policy):
    baseline = artifact(report)
    for trial in trials(baseline):
        trial.update(elapsed_seconds=1.0, total_tokens=10, estimated_spend=0.01)
    head = copy.deepcopy(baseline)
    for trial in trials(head):
        trial.update(elapsed_seconds=2.0, total_tokens=20, estimated_spend=0.02)
    policy.update(
        maximum_median_latency_ratio=1.25,
        maximum_p95_latency_ratio=1.5,
        maximum_median_tokens_ratio=1.25,
        maximum_median_spend_ratio=1.25,
    )
    result = evaluate_gate(policy, head, baseline)
    assert not result["passed"]
    assert (
        len([failure for failure in result["failures"] if "beyond policy" in failure])
        == 4
    )


def test_unknown_spend_never_becomes_zero(report, policy):
    evidence = artifact(report)
    for trial in trials(evidence):
        trial["estimated_spend"] = None
    policy["require_usage"] = True
    assert not evaluate_gate(policy, evidence)["passed"]
    assert evidence["reports"][0]["cases"][CASE]["median_spend"] is None
    policy["require_usage"] = False
    policy["maximum_median_spend_ratio"] = 1.25
    assert not evaluate_gate(policy, evidence, copy.deepcopy(evidence))["passed"]


def test_incompatible_baseline_is_rejected(report, policy):
    baseline = artifact(report)
    head = copy.deepcopy(baseline)
    baseline["reports"][0]["configuration_fingerprint"] = "f" * 64
    assert not evaluate_gate(policy, head, baseline)["passed"]


def test_supplied_aggregates_cannot_hide_failed_trials(report, policy):
    evidence = artifact(report)
    trials(evidence)[0]["passed"] = False
    evidence["reports"][0]["cases"][CASE].update(pass_rate=1, passed=3)
    assert not evaluate_gate(policy, evidence)["passed"]


def test_duplicate_trials_and_duplicate_reports_are_rejected(report, policy):
    evidence = artifact(report)
    trials(evidence)[1]["trial_index"] = 1
    with pytest.raises(ValueError, match="duplicate"):
        evaluate_gate(policy, evidence)
    evidence = artifact(report)
    evidence["reports"].append(copy.deepcopy(evidence["reports"][0]))
    with pytest.raises(ValueError, match="duplicated"):
        evaluate_gate(policy, evidence)


def test_runtime_and_scorer_contradictions_are_rejected(report):
    raw, _ = report
    raw["results"][0]["outcome_status"] = "failed"
    with pytest.raises(ValueError, match="contradicts"):
        validate_reports([raw])


def test_modified_configuration_is_not_accepted_with_old_fingerprint(report):
    raw, _ = report
    raw["configuration"]["max_steps"] += 1
    with pytest.raises(ValueError, match="fingerprint"):
        validate_reports([raw])


def test_missing_declared_scorer_cannot_be_hidden_by_a_passed_outcome(report):
    raw, _ = report
    raw["results"][0]["check_results"] = [
        check
        for check in raw["results"][0]["check_results"]
        if check["name"] == "task_outcome"
    ]
    with pytest.raises(ValueError, match="declared task checks"):
        validate_reports([raw])


def test_cleanup_failure_is_critical_even_with_a_permissive_quality_threshold(
    report, policy
):
    raw, _ = report
    raw["results"][0]["passed"] = False
    raw["results"][0]["failure_reasons"].append(
        {"category": "cleanup_failed", "source": "tool", "message": "cleanup failed"}
    )
    evidence = export_evidence([raw])
    policy["cases"][CASE]["minimum_pass_rate"] = 0
    assert not evaluate_gate(policy, evidence)["passed"]


@pytest.mark.parametrize("name", ["offline", "canary", "release"])
def test_committed_policies_reference_current_catalog(name):
    assert load_policy(name)["cases"]
    assert main(["--policy", name, "--check-policy"]) == 0


def test_policy_typos_unknown_cases_and_nonfinite_values_are_rejected(policy, tmp_path):
    for changes in (
        {"minimum_repeat_count": True},
        {"maximum_spend_ratio": 2},
        {"maximum_pass_rate_drop": float("nan")},
    ):
        with pytest.raises(ValueError):
            validate_policy({**policy, **changes})
    policy["cases"]["missing:built_in_single_agent"] = {"minimum_pass_rate": 1}
    with pytest.raises(ValueError, match="unknown case"):
        validate_policy(policy)
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"a": 1, "a": 2}', encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        strict_json(duplicate)
    duplicate.write_text('{"a": NaN}', encoding="utf-8")
    with pytest.raises(ValueError, match="non-finite"):
        strict_json(duplicate)

from __future__ import annotations

from dataclasses import replace
from threading import Event
from unittest.mock import patch

import pytest

from mikucli.evaluation.bench.catalog import validate_benchmark_tasks
from mikucli.evaluation.bench.errors import BenchmarkError
from mikucli.evaluation.bench.persistence import load_report, result_from_dict
from mikucli.evaluation.bench.reliability import ReliabilityRunner
from mikucli.evaluation.bench.suites import benchmark_cases, benchmark_tasks


@pytest.mark.parametrize("runtime", ["native", "langgraph"])
def test_all_offline_contracts_exercise_real_sessions(runtime, tmp_path):
    results, path, _ = ReliabilityRunner(
        root=tmp_path, runtime=runtime, shuffle_seed=17
    ).run(benchmark_cases("reliability"))
    assert len(results) == 21
    assert all(result.passed for result in results), [
        (result.case_id, result.failure_reasons)
        for result in results
        if not result.passed
    ]
    payload = load_report(path)
    assert payload["status"] == "complete"
    assert len(payload["results"]) == 21
    for raw in payload["results"]:
        restored = result_from_dict(raw)
        assert restored.outcome_status == restored.expected_status
        assert restored.outcome_reason


def test_suite_cancellation_preserves_partial_report_and_resume(tmp_path):
    cases = benchmark_cases("reliability")[:2]
    stop = Event()
    runner = ReliabilityRunner(
        root=tmp_path,
        stop_requested=stop.is_set,
        on_case_finished=lambda _: stop.set(),
    )
    results, path, _ = runner.run(cases)
    assert len(results) == 1
    assert load_report(path)["status"] == "stopped"
    resumed, _, _ = ReliabilityRunner(root=tmp_path, resume_path=path).run(cases)
    assert len(resumed) == 2
    assert load_report(path)["status"] == "complete"


def test_expected_cancellation_does_not_stop_remaining_cases(tmp_path):
    cases = [
        case
        for case in benchmark_cases("reliability")
        if case.task.id
        in {
            "cancel_request",
            "cancel_approval",
            "provider_retry_recovery",
        }
    ]
    results, path, _ = ReliabilityRunner(root=tmp_path).run(cases)
    assert len(results) == 3
    assert all(result.passed for result in results)
    assert load_report(path)["status"] == "complete"


def test_wrong_status_or_failure_reason_fails_even_if_other_checks_pass(tmp_path):
    case = benchmark_cases("reliability")[0]
    for expected in (
        {"expected_status": "failed", "expected_reason": "provider_error"},
        {"expected_reason": "wrong_reason"},
    ):
        wrong = replace(case, task=replace(case.task, **expected))
        result = ReliabilityRunner(root=tmp_path).run_case(wrong)
        assert not result.passed
        assert not next(
            check for check in result.check_results if check.name == "task_outcome"
        ).passed


def test_contract_scorer_detects_bad_side_effects(tmp_path):
    from types import SimpleNamespace

    from mikucli.evaluation.bench.reliability_scenarios import Scenario
    from mikucli.evaluation.bench.reliability_tasks import scenario_invariants

    (tmp_path / "late.txt").write_text("unexpected mutation", encoding="utf-8")
    scenario = Scenario("test", tmp_path, expected_attempts=3, attempts=4)
    messages = scenario_invariants(
        SimpleNamespace(
            workspace=tmp_path,
            setup=SimpleNamespace(metadata={"scenario": scenario}),
        )
    )
    assert any("HTTP attempts" in message for message in messages)
    assert any("work executed" in message for message in messages)


def test_catalog_requires_specific_non_success_outcomes():
    task = benchmark_tasks("reliability")[0]
    with pytest.raises(BenchmarkError, match="expected failure reason"):
        validate_benchmark_tasks(
            [replace(task, expected_status="stopped", expected_reason=None)]
        )
    with pytest.raises(BenchmarkError, match="cannot allow"):
        validate_benchmark_tasks(
            [replace(task, expected_tool_failures=frozenset({"write_file"}))]
        )


def test_offline_cli_never_loads_provider_credentials(tmp_path, capsys):
    from mikucli.evaluation.bench.__main__ import main

    with patch(
        "mikucli.evaluation.bench.__main__.load_config",
        side_effect=AssertionError("credentials loaded"),
    ):
        code = main(
            [
                "--suite",
                "reliability",
                "--workspace",
                str(tmp_path),
                "--case",
                "provider_retry_recovery:built_in_single_agent",
            ]
        )
    assert code == 0
    assert "PASS" in capsys.readouterr().out


def test_public_benchmark_api_dispatches_offline_suite(tmp_path):
    from mikucli.evaluation.bench.runner import run_benchmarks

    results, path, _ = run_benchmarks(
        root=tmp_path,
        client=object(),
        model="unused",
        suite="reliability",
        case_ids={"provider_retry_recovery:built_in_single_agent"},
    )
    assert results[0].passed
    assert load_report(path)["model"] == "offline-contract-v1"


@pytest.mark.parametrize("runtime", ["native", "langgraph"])
def test_deadline_probes_start_after_slow_session_startup(runtime, tmp_path):
    import time

    from mikucli.observability.recorder import LocalTraceRecorder

    original = LocalTraceRecorder.start_trace

    def delayed(recorder, *args, **kwargs):
        time.sleep(0.2)
        return original(recorder, *args, **kwargs)

    cases = [
        case
        for case in benchmark_cases("reliability")
        if case.task.id in {"duration_limit", "mcp_queued_cancellation"}
    ]
    with patch.object(LocalTraceRecorder, "start_trace", new=delayed):
        results, path, _ = ReliabilityRunner(
            root=tmp_path, runtime=runtime, repeat_count=2
        ).run(cases)
    assert len(results) == 4
    assert all(result.passed for result in results), [
        result.failure_reasons for result in results if not result.passed
    ]
    assert load_report(path)["status"] == "complete"

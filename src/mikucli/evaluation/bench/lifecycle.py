from __future__ import annotations

import json
import random
import warnings
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .errors import BenchmarkError
from .metadata import build_run_metadata
from .metrics import read_log_events, summarize_results
from .models import BenchmarkCase, BenchmarkResult
from .persistence import atomic_write_text, load_report, result_from_dict
from .reporting import markdown_report


def run_benchmark_suite(
    runner: Any,
    cases: Iterable[BenchmarkCase] | None,
) -> tuple[list[BenchmarkResult], Path, Path]:
    started_at = datetime.now(timezone.utc)
    selected = list(cases if cases is not None else runner.all_cases())
    if not selected:
        raise BenchmarkError("no benchmark cases selected")
    scheduled = [
        (case, trial_index)
        for trial_index in range(1, runner.repeat_count + 1)
        for case in selected
    ]
    if runner.shuffle_seed is not None:
        random.Random(runner.shuffle_seed).shuffle(scheduled)
    runner.workspaces_root.mkdir(parents=True, exist_ok=True)
    runner.results_root.mkdir(parents=True, exist_ok=True)
    metadata = build_run_metadata(
        root=runner.root,
        model=runner.model,
        max_steps=runner.max_steps,
        context_window_tokens=runner.context_window_tokens,
        cases=selected,
        repeat_count=runner.repeat_count,
        shuffle_seed=runner.shuffle_seed,
        client=runner.client,
        memory_retriever=runner.memory_retriever,
        price=asdict(runner.price) if runner.price is not None else None,
        runtime=runner.runtime,
        run_limits=asdict(runner.run_limits),
    )
    configuration = metadata.get("configuration")
    sandbox = configuration.get("sandbox") if isinstance(configuration, dict) else None
    if runner.require_os_sandbox and not (
        isinstance(sandbox, dict) and sandbox.get("os_enforced") is True
    ):
        raise BenchmarkError(
            "an OS-enforced benchmark sandbox was required but is not available"
        )
    results, resumed_keys, started_at = _load_resume(runner, metadata, started_at)
    planned_trials = len(scheduled)
    checkpoint = _checkpoint_writer(
        runner,
        started_at,
        metadata,
        planned_trials,
        len(selected),
    )
    checkpoint(results, "running")
    stopped = False
    try:
        for case, trial_index in scheduled:
            if (case.id, trial_index) in resumed_keys:
                continue
            if runner.stop_requested():
                stopped = True
                break
            if runner.on_case_started is not None:
                runner.on_case_started(case)
            result = runner.run_case(case, trial_index=trial_index)
            if _trial_was_stopped(result) or (
                result.expected_status == "stopped" and runner.stop_requested()
            ):
                stopped = True
                break
            results.append(result)
            checkpoint(results, "running")
            if runner.on_case_finished is not None:
                runner.on_case_finished(result)
            if runner.stop_requested():
                stopped = True
                break
    except BaseException:
        checkpoint(results, "interrupted")
        raise
    result_path, report_path = checkpoint(results, "stopped" if stopped else "complete")
    return results, result_path, report_path


def _checkpoint_writer(
    runner: Any,
    started_at: datetime,
    metadata: dict[str, object],
    planned_trials: int,
    unique_cases: int,
):
    def write(results: list[BenchmarkResult], status: str) -> tuple[Path, Path]:
        summary = summarize_results(
            results,
            runner.price,
            stopped=status == "stopped",
            planned_trials=planned_trials,
            unique_cases=unique_cases,
            repeat_count=runner.repeat_count,
        )
        summary_payload = asdict(summary)
        suite = metadata.get("suite")
        if isinstance(suite, dict):
            summary_payload["suite_fingerprint"] = suite.get("fingerprint")
        summary_payload["configuration_fingerprint"] = metadata.get(
            "configuration_fingerprint"
        )
        path = runner.resume_path or runner.results_root / f"{runner.run_id}.json"
        report_path = runner.results_root / f"{runner.run_id}.md"
        payload = {
            **metadata,
            "run_id": runner.run_id,
            "run_group_id": runner.run_id,
            "status": status,
            "started_at": started_at.isoformat(),
            "ended_at": datetime.now(timezone.utc).isoformat(),
            "model": runner.model,
            "runtime": runner.runtime,
            "summary": summary_payload,
            "results": [asdict(result) for result in results],
        }
        atomic_write_text(path, json.dumps(payload, indent=2))
        atomic_write_text(
            report_path,
            markdown_report(
                runner.run_id,
                runner.model,
                summary,
                results,
                path,
                runtime=runner.runtime,
                suite_name=(
                    str(suite.get("name") or "core")
                    if isinstance(suite, dict)
                    else "core"
                ),
            ),
        )
        if status != "running":
            try:
                runner.trace_store.import_eval_report(path)
            except Exception as exc:
                warnings.warn(
                    f"benchmark results were written, but observability import failed: {exc}",
                    RuntimeWarning,
                    stacklevel=2,
                )
        return path, report_path

    return write


def _load_resume(
    runner: Any,
    metadata: dict[str, object],
    started_at: datetime,
) -> tuple[list[BenchmarkResult], set[tuple[str, int]], datetime]:
    if runner.resume_path is None:
        return [], set(), started_at
    payload = load_report(runner.resume_path)
    if payload.get("run_id") != runner.run_id:
        raise BenchmarkError("resume report run ID does not match its filename")
    if payload.get("status") == "complete":
        raise BenchmarkError("cannot resume a completed benchmark run")
    if payload.get("suite") != metadata.get("suite"):
        raise BenchmarkError(
            "resume report suite fingerprint or selected cases do not match"
        )
    if payload.get("configuration_fingerprint") != metadata.get(
        "configuration_fingerprint"
    ):
        raise BenchmarkError("resume report configuration does not match this run")
    raw_results = payload.get("results")
    results = (
        [result_from_dict(value) for value in raw_results if isinstance(value, dict)]
        if isinstance(raw_results, list)
        else []
    )
    resumed_keys = {(result.case_id, result.trial_index) for result in results}
    raw_started_at = payload.get("started_at")
    if isinstance(raw_started_at, str):
        try:
            started_at = datetime.fromisoformat(raw_started_at)
        except ValueError:
            pass
    return results, resumed_keys, started_at


def _trial_was_stopped(result: BenchmarkResult) -> bool:
    if result.expected_status == "stopped":
        return False
    return any(
        event.get("type") == "agent_stopped" and event.get("reason") == "user_request"
        for event in read_log_events(Path(result.run_log_path))
    )

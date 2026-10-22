from __future__ import annotations

import shutil
import time
from pathlib import Path

from mikucli.react import SessionResult

from .adapters import (
    BenchmarkConsole,
    RecordingToolSet,
)
from .checks import (
    expected_outcome_check,
    failure_reasons_for_case,
    tool_correctness_checks,
)
from .metrics import (
    cost_from_usage,
    estimate_spend,
    model_retries_from_logs,
    step_retries_from_log,
    trace_id_from_run_log,
)
from .models import (
    ApprovalRecord,
    BenchmarkCase,
    BenchmarkContext,
    BenchmarkMetrics,
    BenchmarkResult,
    CheckResult,
    FailureReason,
    TaskSetup,
)
from .support import (
    safe_case_path,
    snapshot_files,
)


def execute_case(
    runner, case: BenchmarkCase, *, trial_index: int = 1
) -> BenchmarkResult:
    workspace_name = safe_case_path(case.id)
    if runner.repeat_count > 1:
        workspace_name = f"{workspace_name}__trial_{trial_index}"
    workspace = runner.workspaces_root / workspace_name
    _reset_workspace(workspace)
    setup = TaskSetup()
    before_files: dict[str, str] = {}
    after_files: dict[str, str] = {}
    approvals: list[ApprovalRecord] = []
    tool_correctness_results: list[CheckResult] = []
    console = BenchmarkConsole()
    started = time.perf_counter()
    tools: RecordingToolSet | None = None
    session_result = SessionResult(final_answer="", log_path=Path(""))
    check_results: list[CheckResult] = []
    exception_reasons: list[FailureReason] = []
    try:
        setup = case.task.setup(workspace)
        before_files = snapshot_files(workspace)
        base_tools = runner._build_tools(
            workspace=workspace,
            mode=case.session_mode,
            setup=setup,
            approvals=approvals,
            allowed_tools=case.task.allowed_tools,
        )
        tools = RecordingToolSet(base_tools)
        offered_tools = frozenset(
            schema.get("function", {}).get("name")
            for schema in tools.schemas()
            if schema.get("function", {}).get("name")
        )
        session = runner._build_session(
            workspace=workspace,
            mode=case.session_mode,
            tools=tools,
            console=console,
        )
        session_result = session.run_turn(
            case.task.prompt,
            stop_requested=runner.stop_requested,
        )
        check_results.append(expected_outcome_check(case.task, session_result))
        after_files = snapshot_files(workspace)
        final_answer = (
            console.answers[-1] if console.answers else session_result.final_answer
        )
        context = BenchmarkContext(
            workspace=workspace,
            final_answer=final_answer,
            session_result=session_result,
            tool_calls=tools.calls,
            approvals=approvals,
            before_files=before_files,
            after_files=after_files,
            setup=setup,
            offered_tools=offered_tools,
            expected_tool_failures=case.task.expected_tool_failures,
        )
        for check in case.task.checks:
            messages = check(context)
            check_results.append(
                CheckResult(
                    name=getattr(check, "__name__", "check"),
                    passed=not messages,
                    messages=messages,
                    category=getattr(check, "category", "task_success"),
                )
            )
        changed_paths = context.changed_paths
        tool_calls = tools.calls
        final_answer = context.final_answer
        tool_correctness_results = tool_correctness_checks(context, check_results)
    except Exception as exc:  # pragma: no cover - defensive benchmark reporting.
        after_files = snapshot_files(workspace)
        final_answer = f"Benchmark case raised {type(exc).__name__}: {exc}"
        exception_reasons.append(
            FailureReason(category="exception", message=final_answer, source=case.id)
        )
        changed_paths = sorted(
            path
            for path in set(before_files) | set(after_files)
            if before_files.get(path) != after_files.get(path)
        )
        tool_calls = tools.calls if tools is not None else []
    finally:
        if tools is not None:
            try:
                tools.close()
            except Exception as exc:  # Cleanup failures belong in the case report.
                exception_reasons.append(
                    FailureReason(
                        category="cleanup_failed",
                        message=f"Tool cleanup raised {type(exc).__name__}: {exc}",
                        source=case.id,
                    )
                )
    elapsed = time.perf_counter() - started
    elapsed_seconds = round(elapsed, 3)
    passed = all(check.passed for check in [*check_results, *tool_correctness_results])
    if exception_reasons:
        passed = False
    cost = cost_from_usage(console.token_usage_events)
    metrics = BenchmarkMetrics(
        tool_call_count=len(tool_calls),
        model_retries=model_retries_from_logs(workspace, session_result.log_path),
        step_retries=step_retries_from_log(session_result.log_path),
        elapsed_seconds=elapsed_seconds,
        cost=cost,
        price=runner.price,
        estimated_spend=estimate_spend(cost, runner.price),
    )
    failure_reasons = failure_reasons_for_case(
        check_results=check_results,
        tool_calls=tool_calls,
        approvals=approvals,
        final_answer=final_answer,
        run_log_path=session_result.log_path,
    )
    failure_reasons.extend(exception_reasons)
    for signal in tool_correctness_results:
        if signal.passed:
            continue
        for message in signal.messages:
            failure_reasons.append(
                FailureReason(
                    category=signal.category, message=message, source=signal.name
                )
            )
    trace_id = trace_id_from_run_log(session_result.log_path)
    return BenchmarkResult(
        case_id=case.id,
        task_id=case.task.id,
        session_mode=case.session_mode.value,
        passed=passed,
        check_results=check_results,
        final_answer=final_answer,
        changed_paths=changed_paths,
        tool_calls=tool_calls,
        approvals=approvals,
        run_log_path=str(session_result.log_path),
        workspace=str(workspace),
        model=runner.model,
        elapsed_seconds=elapsed_seconds,
        trial_index=trial_index,
        trial_count=runner.repeat_count,
        metrics=metrics,
        failure_reasons=failure_reasons,
        trace_id=trace_id,
        run_group_id=runner.run_id,
        tool_correctness_results=tool_correctness_results,
        outcome_status=session_result.status,
        outcome_reason=session_result.reason,
        expected_status=case.task.expected_status,
    )


def _reset_workspace(workspace: Path) -> None:
    """Start every pending trial from its declared fixture, including after a crash."""
    if workspace.is_symlink():
        workspace.unlink()
    elif workspace.exists():
        shutil.rmtree(workspace)
    workspace.mkdir(parents=True, exist_ok=True)

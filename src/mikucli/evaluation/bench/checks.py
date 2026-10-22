from __future__ import annotations

from pathlib import Path

from mikucli.react import SessionResult

from .metrics import read_log_events
from .models import (
    ApprovalRecord,
    BenchmarkContext,
    BenchmarkTask,
    CheckResult,
    FailureReason,
    ToolCallRecord,
)


def expected_outcome_check(task: BenchmarkTask, result: SessionResult) -> CheckResult:
    matches = result.status == task.expected_status and (
        task.expected_reason is None or result.reason == task.expected_reason
    )
    return CheckResult(
        name="task_outcome",
        passed=matches,
        category="runtime_outcome",
        messages=[]
        if matches
        else [
            f"expected {task.expected_status}/{task.expected_reason or '*'}, "
            f"got {result.status}/{result.reason}"
        ],
    )


def failure_reasons_for_case(
    *,
    check_results: list[CheckResult],
    tool_calls: list[ToolCallRecord],
    approvals: list[ApprovalRecord],
    final_answer: str,
    run_log_path: Path,
) -> list[FailureReason]:
    reasons: list[FailureReason] = []
    for check in check_results:
        if check.passed:
            continue
        for message in check.messages:
            reasons.append(
                FailureReason(
                    category="check_failed", message=message, source=check.name
                )
            )
    for call in tool_calls:
        if not call.ok:
            reasons.append(
                FailureReason(
                    category="tool_failed", message=call.content, source=call.name
                )
            )
    for approval in approvals:
        if not approval.approved:
            reasons.append(
                FailureReason(
                    category="approval_denied",
                    message=approval.summary,
                    source=approval.tool_name,
                )
            )
    if (
        final_answer
        == "Stopped because the session reached the maximum tool loop depth."
    ):
        reasons.append(
            FailureReason(
                category="max_steps_reached", message=final_answer, source="session"
            )
        )
    for event in read_log_events(run_log_path):
        if event.get("type") == "workflow_failed":
            reasons.append(
                FailureReason(
                    category="workflow_failed",
                    message=str(event.get("error") or "orchestrator workflow failed"),
                    source="orchestrator",
                )
            )
    return reasons


def tool_correctness_checks(
    context: BenchmarkContext, task_checks: list[CheckResult]
) -> list[CheckResult]:
    required = sorted(
        check.name.removeprefix("tool_called_")
        for check in task_checks
        if check.name.startswith("tool_called_")
    )
    forbidden = sorted(
        check.name.removeprefix("tool_not_called_")
        for check in task_checks
        if check.name.startswith("tool_not_called_")
    )
    called = [call.name for call in context.tool_calls]
    successful_called = [call.name for call in context.tool_calls if call.ok]
    path_argument_errors = _path_argument_errors(context.tool_calls)
    write_errors = [
        f"write_file changed non-local path: {path}"
        for call in context.tool_calls
        if call.name == "write_file"
        for path in call.changed_paths
        if _path_is_not_workspace_local(path)
    ]
    shell_errors = [
        f"run_shell used platform-suspicious command syntax: {call.arguments.get('command')}"
        for call in context.tool_calls
        if call.name == "run_shell"
        and _shell_command_is_suspicious(str(call.arguments.get("command") or ""))
    ]
    unrecovered_calls = [
        call
        for index, call in enumerate(context.tool_calls)
        if not call.ok
        and call.name in context.offered_tools
        and call.name not in context.expected_tool_failures
        and not any(
            later.name == call.name and later.ok
            for later in context.tool_calls[index + 1 :]
        )
    ]
    high_risk_missing_approval = [
        call.name
        for call in context.tool_calls
        if call.name == "run_shell"
        and (call.ok or call.name not in context.expected_tool_failures)
        and not any(
            approval.tool_name == call.name and approval.approved
            for approval in context.approvals
        )
    ]
    return [
        CheckResult(
            name="required_tools_called",
            category="tool_correctness",
            passed=all(name in successful_called for name in required),
            messages=[
                f"required tool did not complete successfully: {name}"
                for name in required
                if name not in successful_called
            ],
            evidence={
                "required": required,
                "called": called,
                "successful_called": successful_called,
            },
        ),
        CheckResult(
            name="forbidden_tools_not_called",
            category="tool_correctness",
            passed=not any(name in called for name in forbidden),
            messages=[
                f"forbidden tool was called: {name}"
                for name in forbidden
                if name in called
            ],
            evidence={"forbidden": forbidden, "called": called},
        ),
        CheckResult(
            name="tool_arguments_workspace_local",
            category="tool_correctness",
            passed=not path_argument_errors,
            messages=path_argument_errors,
        ),
        CheckResult(
            name="write_file_paths_allowed",
            category="tool_correctness",
            passed=not write_errors,
            messages=write_errors,
        ),
        CheckResult(
            name="run_shell_platform_compatible",
            category="tool_correctness",
            passed=not shell_errors,
            messages=shell_errors,
        ),
        CheckResult(
            name="failed_tools_recovered",
            category="tool_correctness",
            passed=not unrecovered_calls,
            messages=[
                f"failed tool call was not recovered before final answer: {call.name}"
                for call in unrecovered_calls
            ],
            evidence={"unrecovered_failed_tool_count": len(unrecovered_calls)},
        ),
        CheckResult(
            name="high_risk_tools_approved",
            category="tool_correctness",
            passed=not high_risk_missing_approval,
            messages=[
                f"high-risk tool call lacked an approved approval record: {name}"
                for name in high_risk_missing_approval
            ],
        ),
    ]


def _path_argument_errors(tool_calls: list[ToolCallRecord]) -> list[str]:
    errors: list[str] = []
    for call in tool_calls:
        for key, value in call.arguments.items():
            if key not in {"path", "file", "target", "cwd"}:
                continue
            path = str(value)
            if _path_is_not_workspace_local(path):
                errors.append(
                    f"{call.name} argument {key} is not workspace-local: {path}"
                )
    return errors


def _path_is_not_workspace_local(path: str) -> bool:
    stripped = path.strip()
    if not stripped:
        return False
    candidate = Path(stripped)
    lowered = stripped.casefold()
    return (
        candidate.is_absolute()
        or ".." in candidate.parts
        or lowered.startswith("~")
        or "$home" in lowered
        or "%userprofile%" in lowered
    )


def _shell_command_is_suspicious(command: str) -> bool:
    lowered = command.casefold()
    return any(
        token in lowered for token in ("source ", "export ", "rm -rf /", "sudo ")
    )

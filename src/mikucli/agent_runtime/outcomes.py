from __future__ import annotations

from typing import Any

from mikucli.logs import RunLog
from mikucli.run_control import RunLimitExceeded, current_budget

from .contracts import SessionResult


class TaskFailure(RuntimeError):
    def __init__(self, result: SessionResult) -> None:
        super().__init__(result.final_answer)
        self.result = result


def require_success(result: SessionResult) -> None:
    if result.status != "succeeded":
        raise TaskFailure(result)


def trace_outcome(log: RunLog, status: str, attributes: dict[str, Any]) -> str:
    outcome = log.metadata.get("outcome", {})
    attributes.update(
        {"task.status": outcome.get("status"), "task.reason": outcome.get("reason")}
    )
    return (
        "error" if status == "ok" and outcome.get("status") != "succeeded" else status
    )


def finish_session(
    session: Any,
    log: RunLog,
    answer: str,
    trace_status: str,
    error: Exception | None = None,
) -> SessionResult:
    status, reason = "succeeded", "completed"
    if trace_status == "max_steps":
        status, reason = "limit_reached", "max_steps"
    elif trace_status == "stopped":
        status, reason = "stopped", "user_request"
    elif trace_status == "error":
        status, reason = "failed", "workflow_failed"
    if error is not None:
        status, reason = "failed", getattr(error, "reason", "runtime_error")
        if isinstance(error, RunLimitExceeded):
            status = "limit_reached"
        elif isinstance(error, TaskFailure):
            status, reason = error.result.status, error.result.reason
    if status == "succeeded":
        unresolved = {}
        for event in log.events:
            if event["type"] == "tool_result":
                unresolved[event["tool"]] = event
        failures = [event for event in unresolved.values() if not event["ok"]]
        if any(event.get("denied") for event in failures):
            status = "failed"
            reason = "approval_denied"
        elif failures:
            reason = "completed_with_tool_errors"
    budget = current_budget()
    metrics = budget.metrics() if budget is not None else {}
    detail = {"type": type(error).__name__, "message": str(error)} if error else None
    log.final_answer = answer
    log.metadata["outcome"] = {
        "status": status,
        "reason": reason,
        "metrics": metrics,
        "error": detail,
    }
    log.add_event("task_finished", status=status, reason=reason)
    return SessionResult(
        answer,
        session.log_writer.write(log),
        status,
        reason,
        list(log.changed_paths),
        metrics,
        detail,
    )

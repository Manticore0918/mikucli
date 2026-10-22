from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from typing import Any

from mikucli.llm import ToolCall
from mikucli.logs import RunLog
from mikucli.tools.models import ToolResult
from mikucli.run_control import RunLimitExceeded, current_budget

from .cancellation import StopRequested, raise_if_stop_requested
from .messages import capture_tool_output, summarize_value


def handle_tool_calls(session: Any, calls: list[ToolCall], run_log: RunLog, native: bool, **kwargs: Any) -> None:
    completed: set[str] = set()
    try:
        _execute_tool_calls(session, calls, run_log, native, completed=completed, **kwargs)
    except (StopRequested, RunLimitExceeded, KeyboardInterrupt):
        if native:
            for call in calls:
                if call.id not in completed:
                    content = "Tool result unavailable: the task stopped; this tool may have been interrupted or not executed."
                    session.memory.add_tool_result(
                        {"role": "tool", "tool_call_id": call.id, "content": content},
                        tool_name=call.name, ok=False, content=content,
                    )
                    run_log.add_event("tool_skipped", tool=call.name, tool_call_id=call.id, reason="task_stopped")
        raise


def _execute_tool_calls(
    session: Any,
    calls: list[ToolCall],
    run_log: RunLog,
    native: bool,
    *,
    completed: set[str],
    trace_id: str = "",
    parent_span_id: str = "",
    stop_requested: Callable[[], bool] | None = None,
) -> None:
    for call in calls:
        raise_if_stop_requested(stop_requested)
        session.console.tool_request(call.name, call.arguments)
        span_id = session.trace_recorder.start_span(
            trace_id=trace_id,
            name="tool.invoke",
            kind="tool",
            parent_span_id=parent_span_id,
            attributes={
                "tool.name": call.name,
                "tool.call_id": call.id,
                "tool.native": native,
                "tool.arguments": summarize_value(call.arguments),
            },
        )
        invocation_error: Exception | None = None
        try:
            result = session.tools.invoke(call.name, call.arguments)
        except StopRequested as exc:
            invocation_error = exc
            result = _exception_result(call.name, exc)
        except RunLimitExceeded as exc:
            invocation_error = exc
            result = _exception_result(call.name, exc)
        except Exception as exc:  # noqa: BLE001 - normalize extension failures
            invocation_error = exc
            result = _exception_result(call.name, exc)
        budget = current_budget()
        if budget is not None:
            content = result.content.encode("utf-8")
            diff = result.diff.encode("utf-8")
            allowed = budget.output_allowance(len(content) + len(diff))
            if allowed < len(content) + len(diff):
                result = replace(result, ok=False, content=content[:allowed].decode("utf-8", errors="ignore"),
                                 diff=diff[:max(0, allowed - len(content))].decode("utf-8", errors="ignore"))
        span_attributes = {
            "tool.name": call.name,
            "tool.ok": result.ok,
            "tool.changed_paths": result.changed_paths,
            "tool.output": capture_tool_output(result.content),
        }
        if invocation_error is not None:
            span_attributes.update(
                {
                    "error.type": type(invocation_error).__name__,
                    "error.message": str(invocation_error),
                }
            )
        session.trace_recorder.end_span(
            span_id,
            status="ok" if result.ok else "error",
            attributes=span_attributes,
        )
        run_log.add_event(
            "tool_result",
            tool=call.name,
            ok=result.ok,
            denied=result.denied,
            content=result.content,
            changed_paths=result.changed_paths,
        )
        run_log.add_changed_paths(result.changed_paths)
        session.console.tool_result(call.name, result.ok, result.content, result.diff)
        if native:
            session.memory.add_tool_result(
                {
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": result.content,
                },
                tool_name=call.name,
                ok=result.ok,
                content=result.content,
            )
            completed.add(call.id)
        else:
            content = f"Tool result for {call.name}:\n{result.content}"
            session.memory.add_conversation(
                {
                    "role": "user",
                    "content": content,
                },
                content=content,
            )
        if isinstance(invocation_error, (StopRequested, RunLimitExceeded)):
            raise invocation_error
        raise_if_stop_requested(stop_requested)


def _exception_result(tool_name: str, error: Exception) -> ToolResult:
    detail = str(error).strip()
    return ToolResult(
        ok=False,
        content=(
            f"Tool {tool_name} failed unexpectedly ({type(error).__name__})"
            f"{f': {detail}' if detail else ''}."
        ),
    )

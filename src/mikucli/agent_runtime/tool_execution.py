from __future__ import annotations

from collections.abc import Callable
from typing import Any

from mikucli.llm import ToolCall
from mikucli.logs import RunLog

from .cancellation import raise_if_stop_requested
from .messages import capture_tool_output, summarize_value


def handle_tool_calls(
    session: Any,
    calls: list[ToolCall],
    run_log: RunLog,
    native: bool,
    *,
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
        try:
            result = session.tools.invoke(call.name, call.arguments)
        except BaseException as exc:
            session.trace_recorder.end_span(
                span_id,
                status="error",
                attributes={"tool.name": call.name, "error.type": type(exc).__name__, "error.message": str(exc)},
            )
            raise
        session.trace_recorder.end_span(
            span_id,
            status="ok" if result.ok else "error",
            attributes={
                "tool.name": call.name,
                "tool.ok": result.ok,
                "tool.changed_paths": result.changed_paths,
                "tool.output": capture_tool_output(result.content),
            },
        )
        run_log.add_event(
            "tool_result",
            tool=call.name,
            ok=result.ok,
            content=result.content,
            changed_paths=result.changed_paths,
        )
        run_log.add_changed_paths(result.changed_paths)
        session.console.tool_result(call.name, result.ok, result.content, result.diff)
        raise_if_stop_requested(stop_requested)
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
        else:
            content = f"Tool result for {call.name}:\n{result.content}"
            session.memory.add_conversation(
                {
                    "role": "user",
                    "content": content,
                },
                content=content,
            )

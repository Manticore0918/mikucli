from __future__ import annotations

import time
from typing import Any

from mikucli.react import ToolSet
from mikucli.tools import ToolResult

from .models import ChatClient, ToolCallRecord


class BenchmarkConsole:
    """Capture console output emitted while a benchmark case runs."""

    def __init__(self) -> None:
        self.progress_messages: list[str] = []
        self.answers: list[str] = []
        self.token_usage_events: list[Any] = []

    def progress(self, message: str) -> None:
        self.progress_messages.append(message)

    def tool_request(self, name: str, arguments: dict[str, Any]) -> None:
        pass

    def tool_result(self, name: str, ok: bool, content: str, diff: str = "") -> None:
        pass

    def answer(self, content: str) -> None:
        self.answers.append(content)

    def token_usage(self, usage: Any) -> None:
        self.token_usage_events.append(usage)


class RecordingToolSet:
    """Decorate a tool set and retain a compact call history for scoring."""

    def __init__(self, base: ToolSet) -> None:
        self.base = base
        self.calls: list[ToolCallRecord] = []

    def schemas(self) -> list[dict[str, Any]]:
        return self.base.schemas()

    def read_only_tool_names(self) -> set[str]:
        return self.base.read_only_tool_names()

    def requires_approval(self, name: str, arguments: dict[str, Any] | None = None) -> bool:
        return self.base.requires_approval(name, arguments)

    def invoke(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        result = self.base.invoke(name, arguments)
        self.calls.append(
            ToolCallRecord(
                name=name,
                arguments=dict(arguments),
                ok=result.ok,
                content=_compact(result.content),
                changed_paths=list(result.changed_paths),
            )
        )
        return result

    def close(self) -> None:
        close = getattr(self.base, "close", None)
        if callable(close):
            close()


class TimingChatClient:
    """Measure time spent inside provider chat calls."""

    def __init__(self, base: ChatClient) -> None:
        self.base = base
        self.elapsed_seconds = 0.0

    def chat(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        stream: bool = False,
    ) -> Any:
        started = time.perf_counter()
        try:
            return self.base.chat(model=model, messages=messages, tools=tools, stream=stream)
        finally:
            self.elapsed_seconds += time.perf_counter() - started


def _compact(text: str, limit: int = 2000) -> str:
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "..."

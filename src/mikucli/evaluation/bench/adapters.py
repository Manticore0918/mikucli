from __future__ import annotations

from typing import Any

from mikucli.react import ToolSet
from mikucli.tools import ToolRegistry, ToolResult

from .models import ToolCallRecord
from .sandbox import is_safe_benchmark_test_command, run_isolated_unittest


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


class BenchmarkToolRegistry(ToolRegistry):
    """Use the normal built-in tools but replace shell execution with a fixed test run."""

    def run_shell(self, command: str, reason: str, timeout_seconds: int = 30) -> ToolResult:
        del reason
        if not is_safe_benchmark_test_command(command):
            return ToolResult(ok=False, content="benchmark shell command is not an allowed test command.")
        completed = run_isolated_unittest(
            self.workspace.root,
            timeout_seconds=min(timeout_seconds, 30),
        )
        output = (completed.stdout + "\n" + completed.stderr).strip()
        if completed.timed_out:
            return ToolResult(ok=False, content="benchmark tests timed out.\n" + output[-2000:])
        return ToolResult(
            ok=completed.returncode == 0,
            content=output[-4000:] or f"tests exited with code {completed.returncode}",
        )


def _compact(text: str, limit: int = 2000) -> str:
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "..."

from __future__ import annotations

from typing import Any

from mikucli.react import ToolSet
from mikucli.tools import ToolRegistry, ToolResult

from .models import ToolCallRecord
from .sandbox import is_safe_benchmark_test_command, run_isolated_unittest


_BENCHMARK_RUN_SHELL_DESCRIPTION = (
    "Run the benchmark tests from the workspace root. Only a direct pytest or unittest "
    "command is accepted, such as `python -m unittest discover -s tests -v` or "
    "`python -m pytest`. The benchmark makes `src` importable automatically; do not set "
    "environment variables or use shell operators such as `&&`, `|`, `;`, or redirects."
)
_BENCHMARK_RUN_SHELL_REJECTION = (
    "benchmark shell accepts only a direct pytest or unittest command with workspace-relative "
    "arguments. The benchmark makes src importable automatically; do not set PYTHONPATH or use "
    "shell operators such as &&. Retry with `python -m unittest discover -s tests -v` or "
    "`python -m pytest`."
)


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


class RestrictedToolSet:
    """Expose only the tools selected for a benchmark mission."""

    def __init__(self, base: ToolSet, allowed_names: frozenset[str]) -> None:
        self.base = base
        self.allowed_names = allowed_names

    def schemas(self) -> list[dict[str, Any]]:
        return [
            schema
            for schema in self.base.schemas()
            if schema.get("function", {}).get("name") in self.allowed_names
        ]

    def read_only_tool_names(self) -> set[str]:
        return self.base.read_only_tool_names() & self.allowed_names

    def requires_approval(self, name: str, arguments: dict[str, Any] | None = None) -> bool:
        return name in self.allowed_names and self.base.requires_approval(name, arguments)

    def invoke(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        if name not in self.allowed_names:
            return ToolResult(ok=False, content=f"tool is not available for this benchmark mission: {name}")
        return self.base.invoke(name, arguments)

    def stop_current_process(self) -> bool:
        stop = getattr(self.base, "stop_current_process", None)
        return bool(stop()) if callable(stop) else False

    def close(self) -> None:
        close = getattr(self.base, "close", None)
        if callable(close):
            close()


class BenchmarkToolRegistry(ToolRegistry):
    """Use the normal built-in tools but replace shell execution with a fixed test run."""

    def schemas(self) -> list[dict[str, Any]]:
        schemas = super().schemas()
        for schema in schemas:
            function = schema.get("function", {})
            if function.get("name") != "run_shell":
                continue
            function["description"] = _BENCHMARK_RUN_SHELL_DESCRIPTION
            command_schema = function.get("parameters", {}).get("properties", {}).get("command")
            if isinstance(command_schema, dict):
                command_schema["description"] = _BENCHMARK_RUN_SHELL_DESCRIPTION
        return schemas

    def invoke(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        if name == "run_shell" and "command" in arguments:
            command = str(arguments["command"])
            if not is_safe_benchmark_test_command(command):
                return ToolResult(ok=False, content=_BENCHMARK_RUN_SHELL_REJECTION)
        return super().invoke(name, arguments)

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

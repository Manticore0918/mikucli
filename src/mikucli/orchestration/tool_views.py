from __future__ import annotations

from threading import Lock
from typing import Any

from mikucli.llm import TokenUsage
from mikucli.react import Console, ToolSet
from mikucli.tools import ToolResult


class PrefixedConsole:
    """Present subagent output through the shared console with an agent prefix."""

    def __init__(self, console: Console, prefix: str) -> None:
        self.console = console
        self.prefix = prefix

    def progress(self, message: str) -> None:
        if message == "Thinking....":
            return
        self.console.progress(f"{self.prefix}: {message}")

    def tool_request(self, name: str, arguments: dict[str, Any]) -> None:
        self.console.tool_request(f"{self.prefix}.{name}", arguments)

    def tool_result(self, name: str, ok: bool, content: str, diff: str = "") -> None:
        self.console.tool_result(f"{self.prefix}.{name}", ok, content, diff)

    def answer(self, content: str) -> None:
        return

    def token_usage(self, usage: TokenUsage) -> None:
        self.console.token_usage(usage)


class ReadOnlyTools:
    """Expose only the base tool set's explicitly read-only bindings."""

    def __init__(self, base_tools: ToolSet) -> None:
        self.base_tools = base_tools

    def schemas(self) -> list[dict[str, Any]]:
        read_only_names = self.base_tools.read_only_tool_names()
        return [
            schema
            for schema in self.base_tools.schemas()
            if schema.get("function", {}).get("name") in read_only_names
        ]

    def read_only_tool_names(self) -> set[str]:
        return self.base_tools.read_only_tool_names()

    def requires_approval(self, name: str, arguments: dict[str, Any] | None = None) -> bool:
        return self.base_tools.requires_approval(name, arguments)

    def invoke(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        if name in self.base_tools.read_only_tool_names():
            return self.base_tools.invoke(name, arguments)
        return ToolResult(ok=False, content=f"tool is not available in read-only subagent mode: {name}")

    def stop_current_process(self) -> bool:
        stop = getattr(self.base_tools, "stop_current_process", None)
        return bool(stop()) if callable(stop) else False


class SerializedMutationTools:
    """Allow concurrent inspection while serializing approvals and mutations."""

    def __init__(self, base_tools: ToolSet) -> None:
        self.base_tools = base_tools
        self._mutation_lock = Lock()

    def schemas(self) -> list[dict[str, Any]]:
        return self.base_tools.schemas()

    def read_only_tool_names(self) -> set[str]:
        return self.base_tools.read_only_tool_names()

    def requires_approval(self, name: str, arguments: dict[str, Any] | None = None) -> bool:
        checker = getattr(self.base_tools, "requires_approval", None)
        return bool(checker(name, arguments)) if callable(checker) else False

    def invoke(self, name: str, arguments: dict[str, Any]) -> Any:
        if name in self.base_tools.read_only_tool_names() and not self.requires_approval(name, arguments):
            return self.base_tools.invoke(name, arguments)
        with self._mutation_lock:
            return self.base_tools.invoke(name, arguments)

    def stop_current_process(self) -> bool:
        stop = getattr(self.base_tools, "stop_current_process", None)
        return bool(stop()) if callable(stop) else False

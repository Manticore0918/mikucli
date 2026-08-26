from __future__ import annotations

import json
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
        self._denied_calls: set[tuple[str, str]] = set()

    def schemas(self) -> list[dict[str, Any]]:
        return self.base_tools.schemas()

    def read_only_tool_names(self) -> set[str]:
        return self.base_tools.read_only_tool_names()

    def requires_approval(self, name: str, arguments: dict[str, Any] | None = None) -> bool:
        checker = getattr(self.base_tools, "requires_approval", None)
        return bool(checker(name, arguments)) if callable(checker) else False

    def invoke(self, name: str, arguments: dict[str, Any]) -> Any:
        approval_required = self.requires_approval(name, arguments)
        if name in self.base_tools.read_only_tool_names() and not approval_required:
            return self.base_tools.invoke(name, arguments)
        with self._mutation_lock:
            call_key = _tool_call_key(name, arguments)
            if approval_required and call_key in self._denied_calls:
                return ToolResult(
                    ok=False,
                    content=(
                        "tool call not repeated because the user already denied the identical "
                        "approval request in this team workflow."
                    ),
                    denied=True,
                )
            result = self.base_tools.invoke(name, arguments)
            if approval_required and result.denied:
                self._denied_calls.add(call_key)
            return result

    def reset_denied_calls(self) -> None:
        """Start a fresh denial ledger for the next orchestrator turn."""
        with self._mutation_lock:
            self._denied_calls.clear()

    def stop_current_process(self) -> bool:
        stop = getattr(self.base_tools, "stop_current_process", None)
        return bool(stop()) if callable(stop) else False


def _tool_call_key(name: str, arguments: dict[str, Any]) -> tuple[str, str]:
    serialized = json.dumps(arguments, sort_keys=True, separators=(",", ":"), default=str)
    return name, serialized

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


class McpRuntimeError(ValueError):
    pass


@dataclass(frozen=True)
class McpServerStatus:
    name: str
    initialized: bool
    active: bool
    error: str = ""


class McpClient(Protocol):
    def list_tools(self, server_name: str) -> list[Any]: ...
    def call_tool(self, server_name: str, tool_name: str, arguments: dict[str, Any]) -> Any: ...
    def statuses(self) -> list[McpServerStatus]: ...
    def close(self) -> None: ...

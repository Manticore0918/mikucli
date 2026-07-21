from __future__ import annotations

from pathlib import Path
from typing import Any

from ..mcp_config import McpConfig, McpConfigError
from ..tools import ConfirmTool, ToolApprovalRequest, ToolResult, ToolRiskLevel
from .client import ThreadedMcpClient
from .content import field, find_tool, mcp_result_to_tool_result
from .types import McpClient, McpServerStatus


class McpToolSet:
    """Expose configured MCP bindings through the Agent Session ToolSet interface."""

    def __init__(
        self,
        *,
        config: McpConfig,
        client: McpClient,
        workspace: Path,
        confirm_tool: ConfirmTool | None = None,
    ) -> None:
        self.config = config
        self.client = client
        self.workspace = workspace
        self.confirm_tool = confirm_tool
        self._tool_by_binding = self._load_tools_by_binding()
        self._schemas = self._build_schemas()

    @classmethod
    def connect(
        cls,
        *,
        config: McpConfig,
        workspace: Path,
        confirm_tool: ConfirmTool | None = None,
    ) -> McpToolSet:
        client = ThreadedMcpClient(config, workspace)
        try:
            return cls(config=config, client=client, workspace=workspace, confirm_tool=confirm_tool)
        except Exception:
            client.close()
            raise

    def schemas(self) -> list[dict[str, Any]]:
        return list(self._schemas)

    def read_only_tool_names(self) -> set[str]:
        return {name for name, binding in self.config.tools.items() if binding.read_only}

    def requires_approval(self, name: str, arguments: dict[str, Any] | None = None) -> bool:
        binding = self.config.tools.get(name)
        return binding is None or binding.risk != ToolRiskLevel.LOW

    def invoke(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        binding = self.config.tools.get(name)
        if binding is None:
            return ToolResult(ok=False, content=f"unknown MCP tool: {name}")
        approval = ToolApprovalRequest(
            tool_name=name,
            risk_level=binding.risk,
            workspace=str(self.workspace),
            summary=f"Run MCP tool: {name}",
            details=(
                f"server: {binding.server}\n"
                f"mcp_tool_name: {binding.mcp_tool_name}\n"
                f"arguments: {arguments}"
            ),
        )
        if not self._approve(approval):
            return ToolResult(ok=False, content="MCP tool call denied by user.")
        try:
            result = self.client.call_tool(binding.server, binding.mcp_tool_name, arguments)
        except Exception as exc:
            return ToolResult(ok=False, content=f"MCP tool call failed: {exc}")
        return mcp_result_to_tool_result(result)

    def statuses(self) -> list[McpServerStatus]:
        return self.client.statuses()

    def close(self) -> None:
        self.client.close()

    def _load_tools_by_binding(self) -> dict[str, Any]:
        tools_by_binding: dict[str, Any] = {}
        tools_by_server = {
            server_name: self.client.list_tools(server_name)
            for server_name in self.config.servers
        }
        for binding in self.config.tools.values():
            matched = find_tool(tools_by_server[binding.server], binding.mcp_tool_name)
            if matched is None:
                raise McpConfigError(
                    f"MCP tool binding '{binding.model_name}' references missing tool "
                    f"'{binding.mcp_tool_name}' on server '{binding.server}'"
                )
            tools_by_binding[binding.model_name] = matched
        return tools_by_binding

    def _build_schemas(self) -> list[dict[str, Any]]:
        schemas: list[dict[str, Any]] = []
        for binding in self.config.tools.values():
            tool = self._tool_by_binding[binding.model_name]
            schemas.append(
                {
                    "type": "function",
                    "function": {
                        "name": binding.model_name,
                        "description": field(tool, "description") or f"MCP tool {binding.model_name}.",
                        "parameters": field(tool, "inputSchema")
                        or field(tool, "input_schema")
                        or {"type": "object"},
                    },
                }
            )
        return schemas

    def _approve(self, request: ToolApprovalRequest) -> bool:
        if request.risk_level == ToolRiskLevel.LOW:
            return True
        if self.confirm_tool is None:
            return False
        return self.confirm_tool(request)

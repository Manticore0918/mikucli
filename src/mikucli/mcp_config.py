from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .tools import ToolRiskLevel


MCP_CONFIG_PATH = ".mikucli/mcp.json"


class McpConfigError(ValueError):
    pass


@dataclass(frozen=True)
class McpServerConfig:
    name: str
    command: str
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class McpToolBinding:
    model_name: str
    server: str
    mcp_tool_name: str
    risk: ToolRiskLevel = ToolRiskLevel.HIGH

    @property
    def internal_id(self) -> str:
        return f"{self.server}.{self.mcp_tool_name}"


@dataclass(frozen=True)
class McpConfig:
    servers: dict[str, McpServerConfig]
    tools: dict[str, McpToolBinding]


def default_mcp_config_path(workspace: Path) -> Path:
    return workspace / MCP_CONFIG_PATH


def load_mcp_config(workspace: Path, path: Path | None = None) -> McpConfig:
    config_path = path or default_mcp_config_path(workspace)
    if not config_path.exists():
        raise McpConfigError(f"MCP config not found: {config_path}")
    if not config_path.is_file():
        raise McpConfigError(f"MCP config path is not a file: {config_path}")

    try:
        raw = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise McpConfigError(f"invalid MCP config JSON: {exc}") from exc

    if not isinstance(raw, dict):
        raise McpConfigError("MCP config must be a JSON object")
    servers = _parse_servers(raw.get("servers"))
    tools = _parse_tools(raw.get("tools"), servers)
    return McpConfig(servers=servers, tools=tools)


def _parse_servers(raw_servers: Any) -> dict[str, McpServerConfig]:
    if not isinstance(raw_servers, dict) or not raw_servers:
        raise McpConfigError("MCP config must include a non-empty 'servers' object")

    servers: dict[str, McpServerConfig] = {}
    for name, raw_server in raw_servers.items():
        server_name = _require_identifier(str(name), "server name")
        if not isinstance(raw_server, dict):
            raise McpConfigError(f"server '{server_name}' must be an object")
        command = _require_string(raw_server.get("command"), f"server '{server_name}' command")
        args = _string_list(raw_server.get("args", []), f"server '{server_name}' args")
        env = _string_dict(raw_server.get("env", {}), f"server '{server_name}' env")
        servers[server_name] = McpServerConfig(
            name=server_name,
            command=command,
            args=args,
            env=env,
        )
    return servers


def _parse_tools(raw_tools: Any, servers: dict[str, McpServerConfig]) -> dict[str, McpToolBinding]:
    if not isinstance(raw_tools, dict) or not raw_tools:
        raise McpConfigError("MCP config must include a non-empty 'tools' object")

    tools: dict[str, McpToolBinding] = {}
    for model_name, raw_tool in raw_tools.items():
        exposed_name = _require_tool_name(str(model_name), "model-facing tool name")
        if exposed_name in tools:
            raise McpConfigError(f"duplicate MCP model-facing tool name: {exposed_name}")
        if not isinstance(raw_tool, dict):
            raise McpConfigError(f"MCP tool binding '{exposed_name}' must be an object")

        server = _require_identifier(raw_tool.get("server"), f"MCP tool binding '{exposed_name}' server")
        if server not in servers:
            raise McpConfigError(f"MCP tool binding '{exposed_name}' references unknown server: {server}")
        mcp_tool_name = _require_string(
            raw_tool.get("mcp_tool_name"),
            f"MCP tool binding '{exposed_name}' mcp_tool_name",
        )
        risk = _risk(raw_tool.get("risk", "high"), exposed_name)
        tools[exposed_name] = McpToolBinding(
            model_name=exposed_name,
            server=server,
            mcp_tool_name=mcp_tool_name,
            risk=risk,
        )
    return tools


def _require_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise McpConfigError(f"{label} must be a non-empty string")
    return value.strip()


def _require_identifier(value: Any, label: str) -> str:
    parsed = _require_string(value, label)
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", parsed):
        raise McpConfigError(f"{label} contains unsupported characters: {parsed}")
    return parsed


def _require_tool_name(value: str, label: str) -> str:
    parsed = _require_string(value, label)
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", parsed):
        raise McpConfigError(
            f"{label} must be a valid model tool name using letters, numbers, and underscores: {parsed}"
        )
    return parsed


def _string_list(value: Any, label: str) -> list[str]:
    if not isinstance(value, list):
        raise McpConfigError(f"{label} must be a list of strings")
    items: list[str] = []
    for index, item in enumerate(value):
        if not isinstance(item, str):
            raise McpConfigError(f"{label}[{index}] must be a string")
        items.append(item)
    return items


def _string_dict(value: Any, label: str) -> dict[str, str]:
    if not isinstance(value, dict):
        raise McpConfigError(f"{label} must be an object of string values")
    parsed: dict[str, str] = {}
    for key, item in value.items():
        if not isinstance(key, str) or not key:
            raise McpConfigError(f"{label} keys must be non-empty strings")
        if not isinstance(item, str):
            raise McpConfigError(f"{label}.{key} must be a string")
        parsed[key] = item
    return parsed


def _risk(value: Any, model_name: str) -> ToolRiskLevel:
    if not isinstance(value, str):
        raise McpConfigError(f"MCP tool binding '{model_name}' risk must be a string")
    try:
        return ToolRiskLevel(value.strip().lower())
    except ValueError as exc:
        allowed = ", ".join(level.value for level in ToolRiskLevel)
        raise McpConfigError(f"MCP tool binding '{model_name}' risk must be one of: {allowed}") from exc

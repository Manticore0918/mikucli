"""Backward-compatible public facade for MCP runtime capabilities."""

from .mcp_runtime import McpClient, McpRuntimeError, McpServerStatus, McpToolSet, ThreadedMcpClient
from .mcp_runtime.client import run_in_daemon_thread_pool as _run_in_daemon_thread_pool
from .mcp_runtime.content import (
    field as _field,
    find_tool as _find_tool,
    format_mcp_content as _format_mcp_content,
    mcp_result_to_tool_result as _mcp_result_to_tool_result,
)
from .mcp_runtime.server import ConnectedServer as _ConnectedServer
from .mcp_runtime.server import ServerRuntime as _ServerRuntime

__all__ = ["McpClient", "McpRuntimeError", "McpServerStatus", "McpToolSet", "ThreadedMcpClient"]

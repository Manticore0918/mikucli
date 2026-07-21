from __future__ import annotations

from typing import Any

from ..tools import ToolResult


def find_tool(tools: list[Any], name: str) -> Any | None:
    for tool in tools:
        if field(tool, "name") == name:
            return tool
    return None


def field(item: Any, name: str) -> Any:
    if isinstance(item, dict):
        return item.get(name)
    return getattr(item, name, None)


def mcp_result_to_tool_result(result: Any) -> ToolResult:
    is_error = bool(field(result, "isError") or field(result, "is_error"))
    content = format_mcp_content(field(result, "content") or [])
    return ToolResult(ok=not is_error, content=content or ("MCP tool returned an error." if is_error else ""))


def format_mcp_content(blocks: Any) -> str:
    if not isinstance(blocks, list):
        return str(blocks)
    formatted: list[str] = []
    for block in blocks:
        block_type = field(block, "type") or type(block).__name__
        if block_type == "text" or field(block, "text") is not None:
            formatted.append(str(field(block, "text") or ""))
            continue
        metadata = _content_metadata(block)
        formatted.append(f"[MCP {block_type} content: {metadata}]" if metadata else f"[MCP {block_type} content]")
    return "\n".join(part for part in formatted if part)


def _content_metadata(block: Any) -> str:
    parts: list[str] = []
    for name in ("mimeType", "mime_type", "uri", "name"):
        value = field(block, name)
        if value:
            parts.append(f"{name}={value}")
    return ", ".join(parts)

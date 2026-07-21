from __future__ import annotations

import json
import os
from typing import Any

from mikucli.llm import ToolCall


def assistant_message(content: str, tool_calls: list[ToolCall]) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": content,
        "tool_calls": [
            {
                "id": call.id,
                "type": "function",
                "function": {"name": call.name, "arguments": json.dumps(call.arguments)},
            }
            for call in tool_calls
        ],
    }


def capture_tool_output(content: str, *, limit: int = 500) -> str:
    mode = os.environ.get("MIKUCLI_OBS_CAPTURE_TOOL_OUTPUT", "summary").strip().casefold()
    if mode == "off":
        return ""
    if mode == "full" or len(content) <= limit:
        return content
    return content[:limit].rstrip() + "\n... truncated ..."


def summarize_value(value: Any, *, limit: int = 500) -> Any:
    if isinstance(value, dict):
        return {str(key): summarize_value(item, limit=limit) for key, item in value.items()}
    if isinstance(value, list):
        return [summarize_value(item, limit=limit) for item in value]
    if isinstance(value, tuple):
        return [summarize_value(item, limit=limit) for item in value]
    if isinstance(value, str):
        if len(value) <= limit:
            return value
        return value[:limit].rstrip() + "\n... truncated ..."
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    return str(value)

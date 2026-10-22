"""Session-local evidence access works with built-in tools and MCP alike."""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any

from mikucli.tools.models import ToolResult

from .session import SessionMemory
from .storage import MemoryStorageError


class SessionMemoryTools:
    def __init__(self, delegate: Any, memory: SessionMemory) -> None:
        self.delegate, self.memory = delegate, memory

    def __getattr__(self, name: str) -> Any:
        return getattr(self.delegate, name)

    def schemas(self) -> list[dict]:
        own = [
            {
                "type": "function",
                "function": {
                    "name": "recall_session_source",
                    "description": "Read an exact original source from this session, including archived tool arguments. "
                    "Use source IDs in protected state or compressed summaries. Output is paginated.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "source_id": {"type": "string"},
                            "offset": {"type": "integer", "minimum": 0},
                            "limit": {
                                "type": "integer",
                                "minimum": 1,
                                "maximum": 12000,
                            },
                        },
                        "required": ["source_id"],
                        "additionalProperties": False,
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "update_task_state",
                    "description": "Complete or supersede a protected task record with an exact source quote. "
                    "Retiring constraints, decisions, or identifiers requires a later user instruction.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "record_id": {"type": "string"},
                            "status": {
                                "type": "string",
                                "enum": ["active", "completed", "superseded"],
                            },
                            "source_id": {"type": "string"},
                            "quote": {"type": "string"},
                        },
                        "required": ["record_id", "status", "source_id", "quote"],
                        "additionalProperties": False,
                    },
                },
            },
        ]
        base = self.delegate.schemas()
        if {s["function"]["name"] for s in base} & {s["function"]["name"] for s in own}:
            raise ValueError("Tool name conflicts with a reserved session-memory tool.")
        return [*base, *own]

    def invoke(self, name: str, arguments: dict) -> ToolResult:
        if name not in {"recall_session_source", "update_task_state"}:
            return self.delegate.invoke(name, arguments)
        try:
            if name == "recall_session_source":
                if set(arguments) - {"source_id", "offset", "limit"}:
                    raise ValueError("Unknown source lookup arguments.")
                offset, limit = arguments.get("offset", 0), arguments.get("limit", 6000)
                if (
                    type(offset) is not int
                    or type(limit) is not int
                    or offset < 0
                    or not 1 <= limit <= 12000
                ):
                    raise ValueError("Invalid source page bounds.")
                entry = self.memory.get_source(arguments["source_id"])
                text = json.dumps(asdict(entry), ensure_ascii=False)
                return ToolResult(
                    True,
                    json.dumps(
                        {
                            "source_id": entry.entry_id,
                            "offset": offset,
                            "total_chars": len(text),
                            "next_offset": offset + limit
                            if offset + limit < len(text)
                            else None,
                            "content": text[offset : offset + limit],
                        },
                        ensure_ascii=False,
                    ),
                )
            if set(arguments) != {"record_id", "status", "source_id", "quote"}:
                raise ValueError("Expected record_id, status, source_id, and quote.")
            self.memory.update_task_state(**arguments)
            return ToolResult(True, "Task state updated with source-backed evidence.")
        except (ValueError, TypeError, KeyError, MemoryStorageError) as exc:
            return ToolResult(False, str(exc))

    def read_only_tool_names(self) -> set[str]:
        method = getattr(self.delegate, "read_only_tool_names", None)
        return (set(method()) if callable(method) else set()) | {
            "recall_session_source"
        }

    def requires_approval(self, name: str, arguments: dict | None = None) -> bool:
        if name in {"recall_session_source", "update_task_state"}:
            return False
        method = getattr(self.delegate, "requires_approval", None)
        return bool(method(name, arguments)) if callable(method) else False

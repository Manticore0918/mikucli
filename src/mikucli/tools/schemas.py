from __future__ import annotations

from typing import Any


def built_in_tool_schemas(*, include_memory: bool, include_codebase: bool) -> list[dict[str, Any]]:
    """Build model-facing schemas for currently available capabilities."""

    schemas = [
        {
            "type": "function",
            "function": {
                "name": "list_files",
                "description": "List files inside the workspace.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "default": "."},
                        "pattern": {"type": "string", "default": "*"},
                        "max_results": {"type": "integer", "default": 200},
                    },
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": (
                    "Read a UTF-8 text file inside the workspace. Small files can be read in full. "
                    "For large files, use search_codebase to find relevant line numbers, then choose "
                    "optional 1-based inclusive start_line and end_line values for an exact ranged read."
                ),
                "parameters": {
                    "type": "object",
                    "required": ["path"],
                    "properties": {
                        "path": {"type": "string"},
                        "start_line": {
                            "type": "integer",
                            "minimum": 1,
                            "description": "Optional 1-based first line to read (inclusive).",
                        },
                        "end_line": {
                            "type": "integer",
                            "minimum": 1,
                            "description": "Optional 1-based last line to read (inclusive).",
                        },
                    },
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "write_file",
                "description": "Write a UTF-8 text file inside the workspace.",
                "parameters": {
                    "type": "object",
                    "required": ["path", "content"],
                    "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "run_shell",
                "description": (
                    "Run a shell command in the workspace. The command runs from the workspace root. "
                    "On Windows, use cmd.exe-compatible syntax such as "
                    "`set PYTHONPATH=src && python -m unittest discover -s tests`; "
                    "do not use Unix-only commands like `export`, `tail`, `head`, `pwd`, `ls`, or `/workspace`."
                ),
                "parameters": {
                    "type": "object",
                    "required": ["command", "reason"],
                    "properties": {
                        "command": {"type": "string"},
                        "reason": {"type": "string"},
                        "timeout_seconds": {"type": "integer", "default": 30},
                    },
                },
            },
        },
    ]
    if include_memory:
        schemas.append(
            {
                "type": "function",
                "function": {
                    "name": "save_long_term_memory",
                    "description": "Save a durable workspace memory that should be available in future sessions.",
                    "parameters": {
                        "type": "object",
                        "required": ["content"],
                        "properties": {
                            "content": {
                                "type": "string",
                                "description": "A concise fact or preference to remember across sessions.",
                            }
                        },
                    },
                },
            }
        )
    if include_codebase:
        schemas.append(
            {
                "type": "function",
                "function": {
                    "name": "search_codebase",
                    "description": "Search the Codebase Index for relevant workspace source or documentation chunks.",
                    "parameters": {
                        "type": "object",
                        "required": ["query"],
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Natural-language query for codebase retrieval.",
                            },
                            "limit": {"type": "integer", "default": 8},
                        },
                    },
                },
            }
        )
    return schemas

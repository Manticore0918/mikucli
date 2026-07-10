from __future__ import annotations

import fnmatch
import os
import re
import subprocess
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable

from .codebase.formatting import format_search_results
from .codebase.embeddings import EmbeddingError
from .codebase.index import CodebaseIndexError
from .diffing import unified_diff
from .memory import LongTermMemory
from .workspace import Workspace, WorkspaceError


MAX_READ_LINES = 400
MAX_READ_CHARS = 16_000


@dataclass(frozen=True)
class ToolResult:
    ok: bool
    content: str
    changed_paths: list[str] = field(default_factory=list)
    diff: str = ""


class ToolError(ValueError):
    pass


class ToolRiskLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass(frozen=True)
class ToolApprovalRequest:
    tool_name: str
    risk_level: ToolRiskLevel
    workspace: str
    summary: str
    details: str = ""


ConfirmTool = Callable[[ToolApprovalRequest], bool]


class ToolPolicy:
    def __init__(self, risk_levels: dict[str, ToolRiskLevel] | None = None) -> None:
        self.risk_levels = risk_levels or {
            "list_files": ToolRiskLevel.LOW,
            "read_file": ToolRiskLevel.LOW,
            "write_file": ToolRiskLevel.MEDIUM,
            "run_shell": ToolRiskLevel.HIGH,
            "save_long_term_memory": ToolRiskLevel.LOW,
            "search_codebase": ToolRiskLevel.LOW,
        }

    def risk_for(self, tool_name: str) -> ToolRiskLevel:
        try:
            return self.risk_levels[tool_name]
        except KeyError as exc:
            raise ToolError(f"missing tool risk policy for: {tool_name}") from exc


class ToolRegistry:
    def __init__(
        self,
        workspace: Workspace,
        confirm_tool: ConfirmTool | None = None,
        tool_policy: ToolPolicy | None = None,
        long_term_memory: LongTermMemory | None = None,
        codebase_service: Any | None = None,
    ) -> None:
        self.workspace = workspace
        self.confirm_tool = confirm_tool
        self.tool_policy = tool_policy or ToolPolicy()
        self.long_term_memory = long_term_memory
        self.codebase_service = codebase_service

    def schemas(self) -> list[dict[str, Any]]:
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
                        "properties": {
                            "path": {"type": "string"},
                            "content": {"type": "string"},
                        },
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
        if self.long_term_memory is not None:
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
                                },
                            },
                        },
                    },
                }
            )
        if self.codebase_service is not None:
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

    def read_only_tool_names(self) -> set[str]:
        names = {"list_files", "read_file"}
        if self.codebase_service is not None:
            names.add("search_codebase")
        return names

    def requires_approval(self, name: str) -> bool:
        return self.tool_policy.risk_for(name) != ToolRiskLevel.LOW

    def invoke(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        try:
            if name == "list_files":
                return self.list_files(
                    path=str(arguments.get("path", ".")),
                    pattern=str(arguments.get("pattern", "*")),
                    max_results=int(arguments.get("max_results", 200)),
                )
            if name == "read_file":
                return self.read_file(
                    path=str(arguments["path"]),
                    start_line=_optional_int(arguments.get("start_line")),
                    end_line=_optional_int(arguments.get("end_line")),
                )
            if name == "write_file":
                path = str(arguments["path"])
                content = str(arguments["content"])
                approval = self._write_file_approval_request(path, content)
                if not self._approve(approval):
                    return ToolResult(ok=False, content="file change denied by user.", diff=approval.details)
                return self.write_file(path=path, content=content)
            if name == "run_shell":
                command = str(arguments["command"])
                reason = str(arguments["reason"])
                timeout_seconds = int(arguments.get("timeout_seconds", 30))
                if timeout_seconds <= 0 or timeout_seconds > 300:
                    return ToolResult(ok=False, content="timeout_seconds must be between 1 and 300.")
                approval = ToolApprovalRequest(
                    tool_name=name,
                    risk_level=self.tool_policy.risk_for(name),
                    workspace=str(self.workspace.root),
                    summary=f"Run shell command: {command}",
                    details=f"reason: {reason}\ncommand: {command}",
                )
                if not self._approve(approval):
                    return ToolResult(ok=False, content="command denied by user.")
                return self.run_shell(
                    command=command,
                    reason=reason,
                    timeout_seconds=timeout_seconds,
                )
            if name == "save_long_term_memory":
                return self.save_long_term_memory(content=str(arguments["content"]))
            if name == "search_codebase":
                return self.search_codebase(
                    query=str(arguments["query"]),
                    limit=int(arguments.get("limit", 8)),
                )
        except KeyError as exc:
            raise ToolError(f"missing required argument: {exc.args[0]}") from exc
        except ValueError as exc:
            return ToolResult(ok=False, content=str(exc))
        except WorkspaceError as exc:
            return ToolResult(ok=False, content=str(exc))

        raise ToolError(f"unknown tool: {name}")

    def list_files(self, path: str = ".", pattern: str = "*", max_results: int = 200) -> ToolResult:
        root = self.workspace.resolve(path)
        if not root.exists():
            return ToolResult(ok=False, content=f"path does not exist: {path}")
        if not root.is_dir():
            return ToolResult(ok=False, content=f"path is not a directory: {path}")

        matches: list[str] = []
        for item in sorted(root.rglob("*")):
            if len(matches) >= max_results:
                break
            if item.is_file() and not _is_hidden_internal(item, self.workspace.root):
                rel = self.workspace.relative(item)
                if fnmatch.fnmatch(Path(rel).name, pattern) or fnmatch.fnmatch(rel, pattern):
                    matches.append(rel)

        if not matches:
            return ToolResult(ok=True, content="No files matched.")
        return ToolResult(ok=True, content="\n".join(matches))

    def read_file(
        self,
        path: str,
        start_line: int | None = None,
        end_line: int | None = None,
    ) -> ToolResult:
        target = self.workspace.resolve(path)
        if not target.exists():
            return ToolResult(ok=False, content=f"file does not exist: {path}")
        if not target.is_file():
            return ToolResult(ok=False, content=f"path is not a file: {path}")

        content = target.read_text(encoding="utf-8")
        lines = content.splitlines(keepends=True)
        total_lines = len(lines)
        if start_line is None and end_line is None:
            if total_lines <= MAX_READ_LINES and len(content) <= MAX_READ_CHARS:
                return ToolResult(ok=True, content=content)
            return ToolResult(
                ok=False,
                content=_large_file_message(path, total_lines=total_lines, total_chars=len(content)),
            )

        first = start_line if start_line is not None else 1
        last = end_line if end_line is not None else min(total_lines, first + MAX_READ_LINES - 1)
        validation_error = _validate_read_range(first, last, total_lines)
        if validation_error:
            return ToolResult(ok=False, content=validation_error)

        selected = "".join(lines[first - 1 : last])
        selected_line_count = last - first + 1
        if selected_line_count > MAX_READ_LINES or len(selected) > MAX_READ_CHARS:
            return ToolResult(
                ok=False,
                content=(
                    f"requested range {first}-{last} is too large "
                    f"({selected_line_count} lines, {len(selected)} characters). "
                    f"Choose a smaller range of at most {MAX_READ_LINES} lines and {MAX_READ_CHARS} characters."
                ),
            )

        return ToolResult(
            ok=True,
            content=f"File: {path}\nLines: {first}-{last} of {total_lines}\n---\n{selected}",
        )

    def write_file(self, path: str, content: str) -> ToolResult:
        target = self.workspace.resolve(path)
        before = target.read_text(encoding="utf-8") if target.exists() else ""
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        rel = self.workspace.relative(target)
        diff = unified_diff(rel, before, content)
        message = f"Wrote {rel}."
        return ToolResult(ok=True, content=message, changed_paths=[rel], diff=diff)

    def run_shell(self, command: str, reason: str, timeout_seconds: int = 30) -> ToolResult:
        if timeout_seconds <= 0 or timeout_seconds > 300:
            return ToolResult(ok=False, content="timeout_seconds must be between 1 and 300.")

        command, command_env = _extract_leading_env_assignments(command)
        env = _shell_env(self.workspace.root, command_env)
        completed = subprocess.run(
            command,
            cwd=self.workspace.root,
            env=env,
            shell=True,
            text=True,
            capture_output=True,
            timeout=timeout_seconds,
            check=False,
        )
        output = completed.stdout
        if completed.stderr:
            output = f"{output}\n[stderr]\n{completed.stderr}".strip()
        return ToolResult(ok=completed.returncode == 0, content=output or f"exit code {completed.returncode}")

    def save_long_term_memory(self, content: str) -> ToolResult:
        if self.long_term_memory is None:
            return ToolResult(ok=False, content="long-term memory is not available.")

        result = self.long_term_memory.save(content)
        if result.saved:
            return ToolResult(
                ok=True,
                content=f"Saved long-term memory at {result.record.created_at}.",
                changed_paths=[self.workspace.relative(self.long_term_memory.path)],
            )
        return ToolResult(
            ok=True,
            content=f"Long-term memory already exists from {result.record.created_at}.",
        )

    def search_codebase(self, query: str, limit: int = 8) -> ToolResult:
        if self.codebase_service is None:
            return ToolResult(ok=False, content="Codebase Retrieval is not available.")
        if not query.strip():
            return ToolResult(ok=False, content="query cannot be empty.")
        if limit <= 0 or limit > 20:
            return ToolResult(ok=False, content="limit must be between 1 and 20.")
        try:
            results = self.codebase_service.search(query.strip(), limit=limit)
        except (CodebaseIndexError, EmbeddingError) as exc:
            return ToolResult(ok=False, content=str(exc))
        return ToolResult(ok=True, content=format_search_results(results))

    def _write_file_approval_request(self, path: str, content: str) -> ToolApprovalRequest:
        target = self.workspace.resolve(path)
        before = target.read_text(encoding="utf-8") if target.exists() else ""
        rel = self.workspace.relative(target)
        diff = unified_diff(rel, before, content)
        return ToolApprovalRequest(
            tool_name="write_file",
            risk_level=self.tool_policy.risk_for("write_file"),
            workspace=str(self.workspace.root),
            summary=f"Write file: {rel}",
            details=diff,
        )

    def _approve(self, request: ToolApprovalRequest) -> bool:
        if request.risk_level == ToolRiskLevel.LOW:
            return True
        if self.confirm_tool is None:
            return False
        return self.confirm_tool(request)


def _is_hidden_internal(path: Path, workspace_root: Path) -> bool:
    try:
        parts = path.relative_to(workspace_root).parts
    except ValueError:
        return False
    return ".git" in parts or ".mikucli" in parts


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def _validate_read_range(first: int, last: int, total_lines: int) -> str:
    if first < 1:
        return "start_line must be at least 1."
    if last < first:
        return "end_line must be greater than or equal to start_line."
    if total_lines == 0:
        return "cannot select a line range from an empty file."
    if first > total_lines:
        return f"start_line {first} is beyond the end of the file ({total_lines} lines)."
    if last > total_lines:
        return f"end_line {last} is beyond the end of the file ({total_lines} lines)."
    return ""


def _large_file_message(path: str, *, total_lines: int, total_chars: int) -> str:
    return (
        "file is too large for an unbounded read.\n"
        f"Path: {path}\n"
        f"Lines: {total_lines}\n"
        f"Characters: {total_chars}\n"
        f"Maximum per read: {MAX_READ_LINES} lines and {MAX_READ_CHARS} characters.\n"
        "Use search_codebase to locate relevant passages when available, then call read_file again "
        "with 1-based inclusive start_line and end_line values that you choose."
    )


def _extract_leading_env_assignments(command: str) -> tuple[str, dict[str, str]]:
    env: dict[str, str] = {}
    remaining = command.strip()
    while True:
        match = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)=([^\s]+)\s+(.+)$", remaining, flags=re.DOTALL)
        if match is None:
            break
        env[match.group(1)] = match.group(2)
        remaining = match.group(3).strip()
    return remaining, env


def _shell_env(workspace_root: Path, command_env: dict[str, str]) -> dict[str, str]:
    env = os.environ.copy()
    src_path = workspace_root / "src"
    if src_path.is_dir():
        _prepend_env_path(env, "PYTHONPATH", str(src_path))
    for name, value in command_env.items():
        if name == "PYTHONPATH":
            _prepend_env_path(env, name, _normalize_pythonpath(value, workspace_root))
        else:
            env[name] = value
    return env


def _prepend_env_path(env: dict[str, str], name: str, value: str) -> None:
    current = env.get(name)
    env[name] = value if not current else value + os.pathsep + current


def _normalize_pythonpath(value: str, workspace_root: Path) -> str:
    separator = os.pathsep
    parts = value.split(separator)
    normalized: list[str] = []
    for part in parts:
        if not part:
            continue
        path = Path(part)
        normalized.append(str(path if path.is_absolute() else workspace_root / path))
    return separator.join(normalized)

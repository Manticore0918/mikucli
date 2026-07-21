from __future__ import annotations

import fnmatch
import subprocess
from pathlib import Path
from typing import Any

from ..codebase.embeddings import EmbeddingError
from ..codebase.formatting import format_search_results
from ..codebase.index import CodebaseIndexError
from ..diffing import unified_diff
from ..memory import LongTermMemory
from ..sensitive_paths import DEFAULT_SENSITIVE_PATH_POLICY, SensitivePathMatch, SensitivePathPolicy
from ..workspace import Workspace, WorkspaceError
from .helpers import (
    MAX_READ_CHARS,
    MAX_READ_LINES,
    extract_leading_env_assignments,
    is_hidden_internal,
    large_file_message,
    optional_int,
    shell_env,
    validate_read_range,
)
from .models import ConfirmTool, ToolApprovalRequest, ToolError, ToolResult, ToolRiskLevel
from .policy import ToolPolicy
from .schemas import built_in_tool_schemas


class ToolRegistry:
    """Validate and dispatch the built-in tools available to an Agent Session."""

    def __init__(
        self,
        workspace: Workspace,
        confirm_tool: ConfirmTool | None = None,
        tool_policy: ToolPolicy | None = None,
        long_term_memory: LongTermMemory | None = None,
        codebase_service: Any | None = None,
        sensitive_path_policy: SensitivePathPolicy | None = None,
    ) -> None:
        self.workspace = workspace
        self.confirm_tool = confirm_tool
        self.tool_policy = tool_policy or ToolPolicy()
        self.long_term_memory = long_term_memory
        self.codebase_service = codebase_service
        self.sensitive_path_policy = sensitive_path_policy or DEFAULT_SENSITIVE_PATH_POLICY

    def schemas(self) -> list[dict[str, Any]]:
        return built_in_tool_schemas(
            include_memory=self.long_term_memory is not None,
            include_codebase=self.codebase_service is not None,
        )

    def read_only_tool_names(self) -> set[str]:
        names = {"list_files", "read_file"}
        if self.codebase_service is not None:
            names.add("search_codebase")
        return names

    def requires_approval(self, name: str, arguments: dict[str, Any] | None = None) -> bool:
        if self.tool_policy.risk_for(name) != ToolRiskLevel.LOW:
            return True
        if name != "read_file" or not arguments or "path" not in arguments:
            return False
        try:
            target = self.workspace.resolve(str(arguments["path"]))
            rel = self.workspace.relative(target)
        except WorkspaceError:
            return False
        return self.sensitive_path_policy.match(rel) is not None

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
                    start_line=optional_int(arguments.get("start_line")),
                    end_line=optional_int(arguments.get("end_line")),
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
                return self.run_shell(command=command, reason=reason, timeout_seconds=timeout_seconds)
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
            if item.is_file() and not is_hidden_internal(item, self.workspace.root):
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
        rel = self.workspace.relative(target)
        sensitive_match = self.sensitive_path_policy.match(rel)
        if sensitive_match is not None:
            approval = self._read_file_approval_request(rel, sensitive_match)
            if not self._approve(approval):
                return ToolResult(ok=False, content="sensitive file read denied by user.")
        content = target.read_text(encoding="utf-8")
        lines = content.splitlines(keepends=True)
        total_lines = len(lines)
        if start_line is None and end_line is None:
            if total_lines <= MAX_READ_LINES and len(content) <= MAX_READ_CHARS:
                return ToolResult(ok=True, content=content)
            return ToolResult(
                ok=False,
                content=large_file_message(path, total_lines=total_lines, total_chars=len(content)),
            )
        first = start_line if start_line is not None else 1
        last = end_line if end_line is not None else min(total_lines, first + MAX_READ_LINES - 1)
        validation_error = validate_read_range(first, last, total_lines)
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
        return ToolResult(ok=True, content=f"Wrote {rel}.", changed_paths=[rel], diff=diff)

    def run_shell(self, command: str, reason: str, timeout_seconds: int = 30) -> ToolResult:
        if timeout_seconds <= 0 or timeout_seconds > 300:
            return ToolResult(ok=False, content="timeout_seconds must be between 1 and 300.")
        command, command_env = extract_leading_env_assignments(command)
        env = shell_env(self.workspace.root, command_env)
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
        return ToolResult(ok=True, content=f"Long-term memory already exists from {result.record.created_at}.")

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
        results = [result for result in results if self.sensitive_path_policy.match(result.path) is None]
        return ToolResult(ok=True, content=format_search_results(results))

    def _read_file_approval_request(
        self,
        path: str,
        sensitive_match: SensitivePathMatch,
    ) -> ToolApprovalRequest:
        return ToolApprovalRequest(
            tool_name="read_file",
            risk_level=ToolRiskLevel.MEDIUM,
            workspace=str(self.workspace.root),
            summary=f"Read sensitive file: {path}",
            details=(
                f"Sensitive path policy match: {sensitive_match.reason}.\n"
                "No file contents have been read yet."
            ),
        )

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

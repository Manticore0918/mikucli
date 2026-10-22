from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from mikucli.llm import TokenUsage

BASE_AGENT_INSTRUCTIONS = """Use tools only when needed. Do not reveal raw internal reasoning or chain-of-thought.
Only call tools that are actually offered in your available tools; never guess or invent a tool name.
When you need a tool, use native tool calling if available. If native tool calling is not available, emit only strict JSON:
{"tool": "read_file", "arguments": {"path": "README.md"}}
When the user shares a durable preference or fact that should help future sessions, save it with the memory tool if one is available.
When codebase search is available, use it to discover project structure, symbol behavior, implementation locations, or cross-file relationships.
Use read_file after retrieval when you need exact line-level source inspection.
When session-memory tools are offered, use recall_session_source for exact evidence omitted by a summary.
Use update_task_state to complete tasks or blockers only when source evidence supports completion. Keep constraints and decisions active until a later user instruction supersedes them.
When read_file reports that a file is too large, use codebase search to locate relevant passages when available, then choose start_line and end_line for bounded exact reads. Do not retry the same unbounded read.

When you can answer the user, respond normally and concisely.
"""


SYSTEM_PROMPT = f"""You are mikucli, a local command-line agent runner.

{BASE_AGENT_INSTRUCTIONS}
"""


class ToolSet(Protocol):
    def schemas(self) -> list[dict[str, Any]]: ...
    def invoke(self, name: str, arguments: dict[str, Any]) -> Any: ...
    def read_only_tool_names(self) -> set[str]: ...
    def requires_approval(self, name: str, arguments: dict[str, Any] | None = None) -> bool: ...


class Console(Protocol):
    def progress(self, message: str) -> None: ...
    def tool_request(self, name: str, arguments: dict[str, Any]) -> None: ...
    def tool_result(self, name: str, ok: bool, content: str, diff: str = "") -> None: ...
    def answer(self, content: str) -> None: ...
    def token_usage(self, usage: TokenUsage) -> None: ...


@dataclass
class SessionResult:
    final_answer: str
    log_path: Path
    status: str = "succeeded"
    reason: str = "completed"
    changed_paths: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    error: dict[str, str] | None = None

    @property
    def exit_code(self) -> int:
        return {"succeeded": 0, "failed": 1, "stopped": 130, "limit_reached": 3}[self.status]

    def to_dict(self) -> dict[str, Any]:
        return {"schema_version": 1, "status": self.status, "reason": self.reason,
                "final_answer": self.final_answer, "log_path": str(self.log_path),
                "changed_paths": self.changed_paths, "metrics": self.metrics, "error": self.error}

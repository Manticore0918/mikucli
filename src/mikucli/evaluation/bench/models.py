from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Protocol

from mikucli.react import SessionResult, ToolSet
from mikucli.tools import ToolApprovalRequest


class SessionMode(Enum):
    BUILT_IN_SINGLE_AGENT = "built_in_single_agent"
    BUILT_IN_MULTI_AGENT = "built_in_multi_agent"
    MCP_SINGLE_AGENT = "mcp_single_agent"
    MCP_MULTI_AGENT = "mcp_multi_agent"

    @property
    def uses_mcp(self) -> bool:
        return self in {SessionMode.MCP_SINGLE_AGENT, SessionMode.MCP_MULTI_AGENT}

    @property
    def uses_multi_agent(self) -> bool:
        return self in {SessionMode.BUILT_IN_MULTI_AGENT, SessionMode.MCP_MULTI_AGENT}


@dataclass(frozen=True)
class TaskSetup:
    codebase_service: Any | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ToolCallRecord:
    name: str
    arguments: dict[str, Any]
    ok: bool
    content: str
    changed_paths: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ApprovalRecord:
    tool_name: str
    risk_level: str
    summary: str
    details: str
    approved: bool


@dataclass
class BenchmarkContext:
    workspace: Path
    final_answer: str
    session_result: SessionResult
    tool_calls: list[ToolCallRecord]
    approvals: list[ApprovalRecord]
    before_files: dict[str, str]
    after_files: dict[str, str]
    setup: TaskSetup

    @property
    def changed_paths(self) -> list[str]:
        paths = set(self.before_files) | set(self.after_files)
        return sorted(path for path in paths if self.before_files.get(path) != self.after_files.get(path))

    def tool_was_called(self, name: str) -> bool:
        return any(call.name == name for call in self.tool_calls)


Check = Callable[[BenchmarkContext], list[str]]
Setup = Callable[[Path], TaskSetup]


@dataclass(frozen=True)
class BenchmarkTask:
    id: str
    title: str
    prompt: str
    setup: Setup
    checks: tuple[Check, ...]
    session_modes: tuple[SessionMode, ...]


@dataclass(frozen=True)
class BenchmarkCase:
    id: str
    task: BenchmarkTask
    session_mode: SessionMode


@dataclass(frozen=True)
class CheckResult:
    name: str
    passed: bool
    messages: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class EvalCost:
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None


@dataclass(frozen=True)
class EvalPrice:
    prompt_token_price_per_million: float | None = None
    completion_token_price_per_million: float | None = None


@dataclass(frozen=True)
class EstimatedSpend:
    prompt: float | None = None
    completion: float | None = None
    total: float | None = None


@dataclass(frozen=True)
class FailureReason:
    category: str
    message: str
    source: str


@dataclass(frozen=True)
class BenchmarkMetrics:
    tool_call_count: int = 0
    model_retries: int = 0
    step_retries: int = 0
    elapsed_seconds: float = 0.0
    cost: EvalCost = field(default_factory=EvalCost)
    price: EvalPrice | None = None
    estimated_spend: EstimatedSpend | None = None


@dataclass(frozen=True)
class BenchmarkResult:
    case_id: str
    task_id: str
    session_mode: str
    passed: bool
    check_results: list[CheckResult]
    final_answer: str
    changed_paths: list[str]
    tool_calls: list[ToolCallRecord]
    approvals: list[ApprovalRecord]
    run_log_path: str
    workspace: str
    model: str
    elapsed_seconds: float
    metrics: BenchmarkMetrics = field(default_factory=BenchmarkMetrics)
    failure_reasons: list[FailureReason] = field(default_factory=list)


@dataclass(frozen=True)
class BenchmarkRunSummary:
    total_cases: int
    passed_cases: int
    success_rate: float
    tool_call_count: int
    model_retries: int
    step_retries: int
    elapsed_seconds: float
    cost: EvalCost
    price: EvalPrice | None = None
    estimated_spend: EstimatedSpend | None = None
    stopped: bool = False


class ChatClient(Protocol):
    def chat(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        stream: bool = False,
    ) -> Any: ...


class ToolSetFactory(Protocol):
    def __call__(
        self,
        *,
        workspace: Path,
        mode: SessionMode,
        confirm_tool: Callable[[ToolApprovalRequest], bool],
        setup: TaskSetup,
    ) -> ToolSet: ...

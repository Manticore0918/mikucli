from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterable

from mikucli.langgraph_runtime import GraphAgentSession
from mikucli.logs import new_session_id
from mikucli.mcp_config import load_mcp_config
from mikucli.mcp_tools import McpToolSet
from mikucli.memory import MemoryRetriever
from mikucli.multi_agent import OrchestratorSession
from mikucli.observability.recorder import LocalTraceRecorder
from mikucli.observability.store import LocalTraceStore
from mikucli.react import AgentSession, ToolSet
from mikucli.run_control import RunLimits
from mikucli.workspace import Workspace

from .adapters import (
    BenchmarkConsole,
    BenchmarkToolRegistry,
    RecordingToolSet,
    RestrictedToolSet,
)
from .checks import (
    failure_reasons_for_case,
    tool_correctness_checks,
)
from .errors import BenchmarkError
from .execution import execute_case
from .lifecycle import run_benchmark_suite
from .metrics import (
    cost_from_usage,
    estimate_spend,
    model_retries_from_logs,
    step_retries_from_log,
    summarize_results,
    trace_id_from_run_log,
)
from .models import (
    ApprovalRecord,
    BenchmarkCase,
    BenchmarkResult,
    ChatClient,
    EvalPrice,
    SessionMode,
    TaskSetup,
)
from .reporting import markdown_report
from .suites import BenchmarkSuite, benchmark_cases
from .support import (
    approval_recorder,
    observability_store_mode,
    snapshot_files,
)
from .tasks import all_benchmark_cases


class BenchmarkRunner:
    """Coordinate benchmark case execution and persist its artifacts."""

    def __init__(
        self,
        *,
        root: Path,
        client: ChatClient,
        model: str,
        max_steps: int = 30,
        context_window_tokens: int = 128000,
        memory_retriever: MemoryRetriever | None = None,
        price: EvalPrice | None = None,
        stop_requested: Callable[[], bool] | None = None,
        on_case_started: Callable[[BenchmarkCase], None] | None = None,
        on_case_finished: Callable[[BenchmarkResult], None] | None = None,
        repeat_count: int = 1,
        shuffle_seed: int | None = None,
        resume_path: Path | None = None,
        require_os_sandbox: bool = False,
        runtime: str = "native",
        run_limits: RunLimits | None = None,
    ) -> None:
        self.root = root.resolve()
        self.client = client
        self.model = model
        self.max_steps = max_steps
        self.context_window_tokens = context_window_tokens
        self.memory_retriever = memory_retriever
        self.price = price
        self.stop_requested = stop_requested or (lambda: False)
        self.on_case_started = on_case_started
        self.on_case_finished = on_case_finished
        if repeat_count < 1:
            raise BenchmarkError("repeat_count must be at least 1")
        self.repeat_count = repeat_count
        self.shuffle_seed = shuffle_seed
        self.resume_path = resume_path.resolve() if resume_path is not None else None
        self.require_os_sandbox = require_os_sandbox
        if runtime not in {"native", "langgraph"}:
            raise BenchmarkError(f"unsupported agent runtime: {runtime}")
        self.runtime = runtime
        self.run_limits = run_limits or RunLimits()
        self.run_id = (
            self.resume_path.stem if self.resume_path is not None else new_session_id()
        )
        self.bench_root = self.root / ".mikucli" / "evaluation" / "bench"
        self.workspaces_root = self.bench_root / "workspaces" / self.run_id
        self.results_root = self.bench_root / "runs"
        self.trace_store = LocalTraceStore(
            self.root / ".mikucli" / "observability",
            mode=observability_store_mode(),
        )
        self.trace_recorder = LocalTraceRecorder(self.trace_store)

    def run(
        self, cases: Iterable[BenchmarkCase] | None = None
    ) -> tuple[list[BenchmarkResult], Path, Path]:
        return run_benchmark_suite(self, cases)

    @staticmethod
    def all_cases() -> list[BenchmarkCase]:
        return all_benchmark_cases()

    def run_case(self, case: BenchmarkCase, *, trial_index: int = 1) -> BenchmarkResult:
        return execute_case(self, case, trial_index=trial_index)

    def _build_tools(
        self,
        *,
        workspace: Path,
        mode: SessionMode,
        setup: TaskSetup,
        approvals: list[ApprovalRecord],
        allowed_tools: frozenset[str] | None,
    ) -> ToolSet:
        confirm_tool = approval_recorder(workspace, approvals)
        if mode.uses_mcp:
            tools: ToolSet = McpToolSet.connect(
                config=load_mcp_config(workspace),
                workspace=workspace,
                confirm_tool=confirm_tool,
            )
        else:
            tools = BenchmarkToolRegistry(
                workspace=Workspace(workspace),
                confirm_tool=confirm_tool,
                codebase_service=setup.codebase_service,
            )
        if allowed_tools is not None:
            return RestrictedToolSet(tools, allowed_tools)
        return tools

    def _build_session(
        self,
        *,
        workspace: Path,
        mode: SessionMode,
        tools: ToolSet,
        console: BenchmarkConsole,
    ) -> AgentSession | GraphAgentSession | OrchestratorSession:
        kwargs = {
            "client": self.client,
            "model": self.model,
            "workspace": workspace,
            "tools": tools,
            "console": console,
            "max_steps": self.max_steps,
            "context_window_tokens": self.context_window_tokens,
            "memory_retriever": self.memory_retriever,
            "trace_recorder": self.trace_recorder,
            "run_limits": self.run_limits,
        }
        if mode.uses_multi_agent:
            return OrchestratorSession(**kwargs)  # type: ignore[arg-type]
        if self.runtime == "langgraph":
            return GraphAgentSession(**kwargs)  # type: ignore[arg-type]
        return AgentSession(**kwargs)  # type: ignore[arg-type]


def run_benchmarks(
    *,
    root: Path,
    client: ChatClient,
    model: str,
    case_ids: set[str] | None = None,
    max_steps: int = 30,
    context_window_tokens: int = 128000,
    memory_retriever: MemoryRetriever | None = None,
    price: EvalPrice | None = None,
    stop_requested: Callable[[], bool] | None = None,
    on_case_started: Callable[[BenchmarkCase], None] | None = None,
    on_case_finished: Callable[[BenchmarkResult], None] | None = None,
    repeat_count: int = 1,
    shuffle_seed: int | None = None,
    resume_path: Path | None = None,
    require_os_sandbox: bool = False,
    runtime: str = "native",
    suite: BenchmarkSuite = "core",
    run_limits: RunLimits | None = None,
) -> tuple[list[BenchmarkResult], Path, Path]:
    cases = benchmark_cases(suite)
    if case_ids is not None:
        known = {case.id for case in cases}
        unknown = sorted(case_ids - known)
        if unknown:
            raise BenchmarkError("unknown benchmark case(s): " + ", ".join(unknown))
        cases = [case for case in cases if case.id in case_ids]
    if suite == "reliability":
        from .reliability import ReliabilityRunner

        return ReliabilityRunner(
            root=root,
            runtime=runtime,
            repeat_count=repeat_count,
            shuffle_seed=shuffle_seed,
            resume_path=resume_path,
            stop_requested=stop_requested,
            on_case_started=on_case_started,
            on_case_finished=on_case_finished,
            require_os_sandbox=require_os_sandbox,
        ).run(cases)
    return BenchmarkRunner(
        root=root,
        client=client,
        model=model,
        max_steps=max_steps,
        context_window_tokens=context_window_tokens,
        memory_retriever=memory_retriever,
        price=price,
        stop_requested=stop_requested,
        on_case_started=on_case_started,
        on_case_finished=on_case_finished,
        repeat_count=repeat_count,
        shuffle_seed=shuffle_seed,
        resume_path=resume_path,
        require_os_sandbox=require_os_sandbox,
        runtime=runtime,
        run_limits=run_limits,
    ).run(cases)


__all__ = [
    "BenchmarkConsole",
    "BenchmarkError",
    "BenchmarkRunner",
    "RecordingToolSet",
    "cost_from_usage",
    "estimate_spend",
    "failure_reasons_for_case",
    "markdown_report",
    "model_retries_from_logs",
    "run_benchmarks",
    "snapshot_files",
    "step_retries_from_log",
    "summarize_results",
    "tool_correctness_checks",
    "trace_id_from_run_log",
]

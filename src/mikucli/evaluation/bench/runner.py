from __future__ import annotations

import shutil
import time
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
from mikucli.react import AgentSession, SessionResult, ToolSet
from mikucli.workspace import Workspace

from .adapters import BenchmarkConsole, BenchmarkToolRegistry, RecordingToolSet, RestrictedToolSet
from .checks import failure_reasons_for_case, tool_correctness_checks
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
    BenchmarkContext,
    BenchmarkMetrics,
    BenchmarkResult,
    CheckResult,
    ChatClient,
    EvalPrice,
    FailureReason,
    SessionMode,
    TaskSetup,
)
from .lifecycle import run_benchmark_suite
from .errors import BenchmarkError
from .reporting import markdown_report
from .support import approval_recorder, observability_store_mode, safe_case_path, snapshot_files
from .suites import BenchmarkSuite, benchmark_cases
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
        self.run_id = self.resume_path.stem if self.resume_path is not None else new_session_id()
        self.bench_root = self.root / ".mikucli" / "evaluation" / "bench"
        self.workspaces_root = self.bench_root / "workspaces" / self.run_id
        self.results_root = self.bench_root / "runs"
        self.trace_store = LocalTraceStore(
            self.root / ".mikucli" / "observability",
            mode=observability_store_mode(),
        )
        self.trace_recorder = LocalTraceRecorder(self.trace_store)

    def run(self, cases: Iterable[BenchmarkCase] | None = None) -> tuple[list[BenchmarkResult], Path, Path]:
        return run_benchmark_suite(self, cases)

    @staticmethod
    def all_cases() -> list[BenchmarkCase]:
        return all_benchmark_cases()

    def run_case(self, case: BenchmarkCase, *, trial_index: int = 1) -> BenchmarkResult:
        workspace_name = safe_case_path(case.id)
        if self.repeat_count > 1:
            workspace_name = f"{workspace_name}__trial_{trial_index}"
        workspace = self.workspaces_root / workspace_name
        _reset_workspace(workspace)
        setup = TaskSetup()
        before_files: dict[str, str] = {}
        after_files: dict[str, str] = {}
        approvals: list[ApprovalRecord] = []
        tool_correctness_results: list[CheckResult] = []
        console = BenchmarkConsole()
        started = time.perf_counter()
        tools: RecordingToolSet | None = None
        session_result = SessionResult(final_answer="", log_path=Path(""))
        check_results: list[CheckResult] = []
        exception_reasons: list[FailureReason] = []
        try:
            setup = case.task.setup(workspace)
            before_files = snapshot_files(workspace)
            base_tools = self._build_tools(
                workspace=workspace,
                mode=case.session_mode,
                setup=setup,
                approvals=approvals,
                allowed_tools=case.task.allowed_tools,
            )
            tools = RecordingToolSet(base_tools)
            session = self._build_session(
                workspace=workspace,
                mode=case.session_mode,
                tools=tools,
                console=console,
            )
            session_result = session.run_turn(
                case.task.prompt,
                stop_requested=self.stop_requested,
            )
            after_files = snapshot_files(workspace)
            final_answer = console.answers[-1] if console.answers else session_result.final_answer
            context = BenchmarkContext(
                workspace=workspace,
                final_answer=final_answer,
                session_result=session_result,
                tool_calls=tools.calls,
                approvals=approvals,
                before_files=before_files,
                after_files=after_files,
                setup=setup,
            )
            for check in case.task.checks:
                messages = check(context)
                check_results.append(
                    CheckResult(
                        name=getattr(check, "__name__", "check"),
                        passed=not messages,
                        messages=messages,
                        category="task_success",
                    )
                )
            changed_paths = context.changed_paths
            tool_calls = tools.calls
            final_answer = context.final_answer
            tool_correctness_results = tool_correctness_checks(context, check_results)
        except Exception as exc:  # pragma: no cover - defensive benchmark reporting.
            after_files = snapshot_files(workspace)
            final_answer = f"Benchmark case raised {type(exc).__name__}: {exc}"
            exception_reasons.append(
                FailureReason(category="exception", message=final_answer, source=case.id)
            )
            changed_paths = sorted(
                path
                for path in set(before_files) | set(after_files)
                if before_files.get(path) != after_files.get(path)
            )
            tool_calls = tools.calls if tools is not None else []
        finally:
            if tools is not None:
                try:
                    tools.close()
                except Exception as exc:  # Cleanup failures belong in the case report.
                    exception_reasons.append(
                        FailureReason(
                            category="cleanup_failed",
                            message=f"Tool cleanup raised {type(exc).__name__}: {exc}",
                            source=case.id,
                        )
                    )
        elapsed = time.perf_counter() - started
        elapsed_seconds = round(elapsed, 3)
        passed = all(
            check.passed
            for check in [*check_results, *tool_correctness_results]
        )
        if exception_reasons:
            passed = False
        cost = cost_from_usage(console.token_usage_events)
        metrics = BenchmarkMetrics(
            tool_call_count=len(tool_calls),
            model_retries=model_retries_from_logs(workspace, session_result.log_path),
            step_retries=step_retries_from_log(session_result.log_path),
            elapsed_seconds=elapsed_seconds,
            cost=cost,
            price=self.price,
            estimated_spend=estimate_spend(cost, self.price),
        )
        failure_reasons = failure_reasons_for_case(
            check_results=check_results,
            tool_calls=tool_calls,
            approvals=approvals,
            final_answer=final_answer,
            run_log_path=session_result.log_path,
        )
        failure_reasons.extend(exception_reasons)
        for signal in tool_correctness_results:
            if signal.passed:
                continue
            for message in signal.messages:
                failure_reasons.append(
                    FailureReason(category=signal.category, message=message, source=signal.name)
                )
        trace_id = trace_id_from_run_log(session_result.log_path)
        return BenchmarkResult(
            case_id=case.id,
            task_id=case.task.id,
            session_mode=case.session_mode.value,
            passed=passed,
            check_results=check_results,
            final_answer=final_answer,
            changed_paths=changed_paths,
            tool_calls=tool_calls,
            approvals=approvals,
            run_log_path=str(session_result.log_path),
            workspace=str(workspace),
            model=self.model,
            elapsed_seconds=elapsed_seconds,
            trial_index=trial_index,
            trial_count=self.repeat_count,
            metrics=metrics,
            failure_reasons=failure_reasons,
            trace_id=trace_id,
            run_group_id=self.run_id,
            tool_correctness_results=tool_correctness_results,
        )

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
        }
        if mode.uses_multi_agent:
            return OrchestratorSession(**kwargs)  # type: ignore[arg-type]
        if self.runtime == "langgraph":
            return GraphAgentSession(**kwargs)  # type: ignore[arg-type]
        return AgentSession(**kwargs)  # type: ignore[arg-type]


def _reset_workspace(workspace: Path) -> None:
    """Start every pending trial from its declared fixture, including after a crash."""
    if workspace.is_symlink():
        workspace.unlink()
    elif workspace.exists():
        shutil.rmtree(workspace)
    workspace.mkdir(parents=True, exist_ok=True)


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
) -> tuple[list[BenchmarkResult], Path, Path]:
    cases = benchmark_cases(suite)
    if case_ids is not None:
        known = {case.id for case in cases}
        unknown = sorted(case_ids - known)
        if unknown:
            raise BenchmarkError("unknown benchmark case(s): " + ", ".join(unknown))
        cases = [case for case in cases if case.id in case_ids]
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

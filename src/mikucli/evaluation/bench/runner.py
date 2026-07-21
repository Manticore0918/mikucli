from __future__ import annotations

import json
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable

from mikucli.logs import new_session_id
from mikucli.mcp_config import load_mcp_config
from mikucli.mcp_tools import McpToolSet
from mikucli.multi_agent import OrchestratorSession
from mikucli.observability.recorder import LocalTraceRecorder
from mikucli.observability.store import LocalTraceStore
from mikucli.react import AgentSession, SessionResult, ToolSet
from mikucli.tools import ToolRegistry
from mikucli.workspace import Workspace

from .adapters import BenchmarkConsole, RecordingToolSet, TimingChatClient
from .checks import failure_reasons_for_case, hallucination_checks, tool_correctness_checks
from .metrics import (
    cost_from_usage,
    estimate_spend,
    model_retries,
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
    BenchmarkRunSummary,
    CheckResult,
    ChatClient,
    EvalPrice,
    FailureReason,
    SessionMode,
    TaskSetup,
)
from .reporting import markdown_report
from .support import approval_recorder, observability_store_mode, safe_case_path, snapshot_files
from .tasks import all_benchmark_cases


class BenchmarkError(ValueError):
    pass


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
        price: EvalPrice | None = None,
        stop_requested: Callable[[], bool] | None = None,
        on_case_started: Callable[[BenchmarkCase], None] | None = None,
        on_case_finished: Callable[[BenchmarkResult], None] | None = None,
    ) -> None:
        self.root = root.resolve()
        self.client = client
        self.model = model
        self.max_steps = max_steps
        self.context_window_tokens = context_window_tokens
        self.price = price
        self.stop_requested = stop_requested or (lambda: False)
        self.on_case_started = on_case_started
        self.on_case_finished = on_case_finished
        self.run_id = new_session_id()
        self.bench_root = self.root / ".mikucli" / "evaluation" / "bench"
        self.workspaces_root = self.bench_root / "workspaces" / self.run_id
        self.results_root = self.bench_root / "runs"
        self.trace_store = LocalTraceStore(
            self.root / ".mikucli" / "observability",
            mode=observability_store_mode(),
        )
        self.trace_recorder = LocalTraceRecorder(self.trace_store)

    def run(self, cases: Iterable[BenchmarkCase] | None = None) -> tuple[list[BenchmarkResult], Path, Path]:
        selected = list(cases if cases is not None else all_benchmark_cases())
        if not selected:
            raise BenchmarkError("no benchmark cases selected")
        self.workspaces_root.mkdir(parents=True, exist_ok=True)
        self.results_root.mkdir(parents=True, exist_ok=True)
        results: list[BenchmarkResult] = []
        stopped = False
        for case in selected:
            if self.stop_requested():
                stopped = True
                break
            if self.on_case_started is not None:
                self.on_case_started(case)
            result = self.run_case(case)
            results.append(result)
            if self.on_case_finished is not None:
                self.on_case_finished(result)
            if self.stop_requested():
                stopped = True
                break
        summary = summarize_results(results, self.price, stopped=stopped)
        result_path, report_path = self._write_results(results, summary)
        return results, result_path, report_path

    def run_case(self, case: BenchmarkCase) -> BenchmarkResult:
        workspace = self.workspaces_root / safe_case_path(case.id)
        workspace.mkdir(parents=True, exist_ok=True)
        setup = TaskSetup()
        before_files: dict[str, str] = {}
        after_files: dict[str, str] = {}
        approvals: list[ApprovalRecord] = []
        hallucination_results: list[CheckResult] = []
        tool_correctness_results: list[CheckResult] = []
        console = BenchmarkConsole()
        started = time.perf_counter()
        timing_client = TimingChatClient(self.client)
        tools: RecordingToolSet | None = None
        session_result = SessionResult(final_answer="", log_path=Path(""))
        check_results: list[CheckResult] = []
        exception_reason: FailureReason | None = None
        try:
            setup = case.task.setup(workspace)
            before_files = snapshot_files(workspace)
            base_tools = self._build_tools(
                workspace=workspace,
                mode=case.session_mode,
                setup=setup,
                approvals=approvals,
            )
            tools = RecordingToolSet(base_tools)
            session = self._build_session(
                workspace=workspace,
                mode=case.session_mode,
                tools=tools,
                console=console,
                client=timing_client,
            )
            session_result = session.run_turn(case.task.prompt)
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
            hallucination_results = hallucination_checks(context, check_results)
            tool_correctness_results = tool_correctness_checks(context, check_results)
        except Exception as exc:  # pragma: no cover - defensive benchmark reporting.
            after_files = snapshot_files(workspace)
            final_answer = f"Benchmark case raised {type(exc).__name__}: {exc}"
            exception_reason = FailureReason(category="exception", message=final_answer, source=case.id)
            changed_paths = sorted(
                path
                for path in set(before_files) | set(after_files)
                if before_files.get(path) != after_files.get(path)
            )
            tool_calls = tools.calls if tools is not None else []
        finally:
            if tools is not None:
                tools.close()
        elapsed = time.perf_counter() - started
        elapsed_seconds = round(elapsed, 3)
        llm_latency_seconds = round(timing_client.elapsed_seconds, 3)
        agent_latency_seconds = round(max(0.0, elapsed - timing_client.elapsed_seconds), 3)
        passed = all(check.passed for check in [*check_results, *hallucination_results])
        if exception_reason is not None:
            passed = False
        cost = cost_from_usage(console.token_usage_events)
        metrics = BenchmarkMetrics(
            tool_call_count=len(tool_calls),
            model_retries=model_retries(tool_calls, final_answer),
            step_retries=step_retries_from_log(session_result.log_path),
            elapsed_seconds=elapsed_seconds,
            agent_latency_seconds=agent_latency_seconds,
            llm_latency_seconds=llm_latency_seconds,
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
        if exception_reason is not None:
            failure_reasons.append(exception_reason)
        for signal in [*hallucination_results, *tool_correctness_results]:
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
            metrics=metrics,
            failure_reasons=failure_reasons,
            trace_id=trace_id,
            run_group_id=self.run_id,
            hallucination_results=hallucination_results,
            tool_correctness_results=tool_correctness_results,
        )

    def _build_tools(
        self,
        *,
        workspace: Path,
        mode: SessionMode,
        setup: TaskSetup,
        approvals: list[ApprovalRecord],
    ) -> ToolSet:
        confirm_tool = approval_recorder(workspace, approvals)
        if mode.uses_mcp:
            return McpToolSet.connect(
                config=load_mcp_config(workspace),
                workspace=workspace,
                confirm_tool=confirm_tool,
            )
        return ToolRegistry(
            workspace=Workspace(workspace),
            confirm_tool=confirm_tool,
            codebase_service=setup.codebase_service,
        )

    def _build_session(
        self,
        *,
        workspace: Path,
        mode: SessionMode,
        tools: ToolSet,
        console: BenchmarkConsole,
        client: ChatClient | None = None,
    ) -> AgentSession | OrchestratorSession:
        kwargs = {
            "client": client or self.client,
            "model": self.model,
            "workspace": workspace,
            "tools": tools,
            "console": console,
            "max_steps": self.max_steps,
            "context_window_tokens": self.context_window_tokens,
            "trace_recorder": self.trace_recorder,
        }
        if mode.uses_multi_agent:
            return OrchestratorSession(**kwargs)  # type: ignore[arg-type]
        return AgentSession(**kwargs)  # type: ignore[arg-type]

    def _write_results(
        self,
        results: list[BenchmarkResult],
        summary: BenchmarkRunSummary,
    ) -> tuple[Path, Path]:
        path = self.results_root / f"{self.run_id}.json"
        report_path = self.results_root / f"{self.run_id}.md"
        payload = {
            "run_id": self.run_id,
            "run_group_id": self.run_id,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "model": self.model,
            "summary": asdict(summary),
            "results": [asdict(result) for result in results],
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        report_path.write_text(
            markdown_report(self.run_id, self.model, summary, results, path),
            encoding="utf-8",
        )
        self.trace_store.import_eval_report(path)
        return path, report_path


def run_benchmarks(
    *,
    root: Path,
    client: ChatClient,
    model: str,
    case_ids: set[str] | None = None,
    max_steps: int = 30,
    context_window_tokens: int = 128000,
    price: EvalPrice | None = None,
    stop_requested: Callable[[], bool] | None = None,
    on_case_started: Callable[[BenchmarkCase], None] | None = None,
    on_case_finished: Callable[[BenchmarkResult], None] | None = None,
) -> tuple[list[BenchmarkResult], Path, Path]:
    cases = all_benchmark_cases()
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
        price=price,
        stop_requested=stop_requested,
        on_case_started=on_case_started,
        on_case_finished=on_case_finished,
    ).run(cases)


__all__ = [
    "BenchmarkConsole",
    "BenchmarkError",
    "BenchmarkRunner",
    "RecordingToolSet",
    "TimingChatClient",
    "cost_from_usage",
    "estimate_spend",
    "failure_reasons_for_case",
    "hallucination_checks",
    "markdown_report",
    "model_retries",
    "run_benchmarks",
    "snapshot_files",
    "step_retries_from_log",
    "summarize_results",
    "tool_correctness_checks",
    "trace_id_from_run_log",
]

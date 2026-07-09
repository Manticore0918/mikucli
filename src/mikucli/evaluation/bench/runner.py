from __future__ import annotations

import json
import re
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

from mikucli.logs import new_session_id
from mikucli.mcp_config import load_mcp_config
from mikucli.mcp_tools import McpToolSet
from mikucli.multi_agent import OrchestratorSession
from mikucli.react import AgentSession, SessionResult, ToolSet
from mikucli.tools import ToolApprovalRequest, ToolRegistry, ToolResult
from mikucli.workspace import Workspace

from .tasks import all_benchmark_cases
from .models import (
    ApprovalRecord,
    BenchmarkCase,
    BenchmarkContext,
    BenchmarkMetrics,
    BenchmarkResult,
    BenchmarkRunSummary,
    CheckResult,
    ChatClient,
    EstimatedSpend,
    EvalCost,
    EvalPrice,
    FailureReason,
    SessionMode,
    TaskSetup,
    ToolCallRecord,
)


class BenchmarkError(ValueError):
    pass


class BenchmarkConsole:
    def __init__(self) -> None:
        self.progress_messages: list[str] = []
        self.answers: list[str] = []
        self.token_usage_events: list[Any] = []

    def progress(self, message: str) -> None:
        self.progress_messages.append(message)

    def tool_request(self, name: str, arguments: dict[str, Any]) -> None:
        pass

    def tool_result(self, name: str, ok: bool, content: str, diff: str = "") -> None:
        pass

    def answer(self, content: str) -> None:
        self.answers.append(content)

    def token_usage(self, usage: Any) -> None:
        self.token_usage_events.append(usage)


class RecordingToolSet:
    def __init__(self, base: ToolSet) -> None:
        self.base = base
        self.calls: list[ToolCallRecord] = []

    def schemas(self) -> list[dict[str, Any]]:
        return self.base.schemas()

    def read_only_tool_names(self) -> set[str]:
        return self.base.read_only_tool_names()

    def invoke(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        result = self.base.invoke(name, arguments)
        self.calls.append(
            ToolCallRecord(
                name=name,
                arguments=dict(arguments),
                ok=result.ok,
                content=_compact(result.content),
                changed_paths=list(result.changed_paths),
            )
        )
        return result

    def close(self) -> None:
        close = getattr(self.base, "close", None)
        if callable(close):
            close()


class TimingChatClient:
    def __init__(self, base: ChatClient) -> None:
        self.base = base
        self.elapsed_seconds = 0.0

    def chat(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        stream: bool = False,
    ) -> Any:
        started = time.perf_counter()
        try:
            return self.base.chat(model=model, messages=messages, tools=tools, stream=stream)
        finally:
            self.elapsed_seconds += time.perf_counter() - started


class BenchmarkRunner:
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
        workspace = self.workspaces_root / _safe_case_path(case.id)
        workspace.mkdir(parents=True, exist_ok=True)
        setup = TaskSetup()
        before_files: dict[str, str] = {}
        after_files: dict[str, str] = {}
        approvals: list[ApprovalRecord] = []
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
                check_results.append(CheckResult(name=getattr(check, "__name__", "check"), passed=not messages, messages=messages))
            changed_paths = context.changed_paths
            tool_calls = tools.calls
            final_answer = context.final_answer
        except Exception as exc:  # pragma: no cover - exercised through defensive benchmark reporting.
            after_files = snapshot_files(workspace)
            final_answer = f"Benchmark case raised {type(exc).__name__}: {exc}"
            exception_reason = FailureReason(
                category="exception",
                message=final_answer,
                source=case.id,
            )
            changed_paths = sorted(path for path in set(before_files) | set(after_files) if before_files.get(path) != after_files.get(path))
            tool_calls = tools.calls if tools is not None else []
        finally:
            if tools is not None:
                tools.close()
        elapsed = time.perf_counter() - started
        elapsed_seconds = round(elapsed, 3)
        llm_latency_seconds = round(timing_client.elapsed_seconds, 3)
        agent_latency_seconds = round(max(0.0, elapsed - timing_client.elapsed_seconds), 3)
        passed = all(check.passed for check in check_results)
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
        )

    def _build_tools(
        self,
        *,
        workspace: Path,
        mode: SessionMode,
        setup: TaskSetup,
        approvals: list[ApprovalRecord],
    ) -> ToolSet:
        confirm_tool = _approval_recorder(workspace, approvals)
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
        }
        if mode.uses_multi_agent:
            return OrchestratorSession(**kwargs)  # type: ignore[arg-type]
        return AgentSession(**kwargs)  # type: ignore[arg-type]

    def _write_results(self, results: list[BenchmarkResult], summary: BenchmarkRunSummary) -> tuple[Path, Path]:
        path = self.results_root / f"{self.run_id}.json"
        report_path = self.results_root / f"{self.run_id}.md"
        payload = {
            "run_id": self.run_id,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "model": self.model,
            "summary": asdict(summary),
            "results": [asdict(result) for result in results],
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        report_path.write_text(markdown_report(self.run_id, self.model, summary, results, path), encoding="utf-8")
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


def summarize_results(
    results: list[BenchmarkResult],
    price: EvalPrice | None = None,
    *,
    stopped: bool = False,
) -> BenchmarkRunSummary:
    total = len(results)
    passed = sum(1 for result in results if result.passed)
    cost = sum_costs([result.metrics.cost for result in results])
    elapsed = round(sum(result.metrics.elapsed_seconds for result in results), 3)
    agent_latency = round(sum(result.metrics.agent_latency_seconds for result in results), 3)
    llm_latency = round(sum(result.metrics.llm_latency_seconds for result in results), 3)
    return BenchmarkRunSummary(
        total_cases=total,
        passed_cases=passed,
        success_rate=round(passed / total, 4) if total else 0.0,
        tool_call_count=sum(result.metrics.tool_call_count for result in results),
        model_retries=sum(result.metrics.model_retries for result in results),
        step_retries=sum(result.metrics.step_retries for result in results),
        elapsed_seconds=elapsed,
        agent_latency_seconds=agent_latency,
        llm_latency_seconds=llm_latency,
        cost=cost,
        price=price,
        estimated_spend=estimate_spend(cost, price),
        stopped=stopped,
    )


def cost_from_usage(events: list[Any]) -> EvalCost:
    prompt_values = [event.prompt_tokens for event in events if event.prompt_tokens is not None]
    completion_values = [event.completion_tokens for event in events if event.completion_tokens is not None]
    total_values = [event.total_tokens for event in events if event.total_tokens is not None]
    return EvalCost(
        prompt_tokens=sum(prompt_values) if prompt_values else None,
        completion_tokens=sum(completion_values) if completion_values else None,
        total_tokens=sum(total_values) if total_values else None,
    )


def sum_costs(costs: list[EvalCost]) -> EvalCost:
    prompt_values = [cost.prompt_tokens for cost in costs if cost.prompt_tokens is not None]
    completion_values = [cost.completion_tokens for cost in costs if cost.completion_tokens is not None]
    total_values = [cost.total_tokens for cost in costs if cost.total_tokens is not None]
    return EvalCost(
        prompt_tokens=sum(prompt_values) if prompt_values else None,
        completion_tokens=sum(completion_values) if completion_values else None,
        total_tokens=sum(total_values) if total_values else None,
    )


def estimate_spend(cost: EvalCost, price: EvalPrice | None) -> EstimatedSpend | None:
    if price is None:
        return None
    prompt = _component_spend(cost.prompt_tokens, price.prompt_token_price_per_million)
    completion = _component_spend(cost.completion_tokens, price.completion_token_price_per_million)
    total = round(prompt + completion, 8) if prompt is not None and completion is not None else None
    return EstimatedSpend(prompt=prompt, completion=completion, total=total)


def model_retries(tool_calls: list[ToolCallRecord], final_answer: str) -> int:
    retries = sum(1 for call in tool_calls if not call.ok)
    if final_answer == "Stopped because the session reached the maximum tool loop depth.":
        retries += 1
    return retries


def step_retries_from_log(path: Path) -> int:
    events = _read_log_events(path)
    attempts_by_step: dict[str, int] = {}
    for event in events:
        if event.get("type") != "step_worker_result":
            continue
        step_id = str(event.get("step_id") or "")
        if not step_id:
            continue
        try:
            attempt = int(event.get("attempt") or 0)
        except (TypeError, ValueError):
            continue
        attempts_by_step[step_id] = max(attempts_by_step.get(step_id, 0), attempt)
    return sum(max(0, attempts - 1) for attempts in attempts_by_step.values())


def failure_reasons_for_case(
    *,
    check_results: list[CheckResult],
    tool_calls: list[ToolCallRecord],
    approvals: list[ApprovalRecord],
    final_answer: str,
    run_log_path: Path,
) -> list[FailureReason]:
    reasons: list[FailureReason] = []
    for check in check_results:
        if check.passed:
            continue
        for message in check.messages:
            reasons.append(FailureReason(category="check_failed", message=message, source=check.name))
    for call in tool_calls:
        if not call.ok:
            reasons.append(FailureReason(category="tool_failed", message=call.content, source=call.name))
    for approval in approvals:
        if not approval.approved:
            reasons.append(
                FailureReason(
                    category="approval_denied",
                    message=approval.summary,
                    source=approval.tool_name,
                )
            )
    if final_answer == "Stopped because the session reached the maximum tool loop depth.":
        reasons.append(FailureReason(category="max_steps_reached", message=final_answer, source="session"))
    for event in _read_log_events(run_log_path):
        if event.get("type") == "workflow_failed":
            reasons.append(
                FailureReason(
                    category="workflow_failed",
                    message=str(event.get("error") or "orchestrator workflow failed"),
                    source="orchestrator",
                )
            )
    return reasons


def markdown_report(
    run_id: str,
    model: str,
    summary: BenchmarkRunSummary,
    results: list[BenchmarkResult],
    json_path: Path,
) -> str:
    lines = [
        f"# Benchmark Run {run_id}",
        "",
        f"- Model: `{model}`",
        f"- JSON results: `{json_path}`",
        f"- Success rate: {_fmt_percent(summary.success_rate)} ({summary.passed_cases}/{summary.total_cases})",
        f"- Stopped: {'yes' if summary.stopped else 'no'}",
        f"- Tool calls: {summary.tool_call_count}",
        f"- Model retries: {summary.model_retries}",
        f"- Step retries: {summary.step_retries}",
        f"- Total latency: {summary.elapsed_seconds:.3f}s",
        f"- Agent latency: {summary.agent_latency_seconds:.3f}s",
        f"- LLM latency: {summary.llm_latency_seconds:.3f}s",
        f"- Cost: {_fmt_cost(summary.cost)}",
        f"- Estimated spend: {_fmt_spend(summary.estimated_spend)}",
        "",
        "## Cases",
        "",
        "| Status | Case | Mode | Tool calls | Model retries | Step retries | Total latency | Agent latency | LLM latency | Cost | Spend |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for result in results:
        status = "PASS" if result.passed else "FAIL"
        lines.append(
            "| "
            + " | ".join(
                [
                    status,
                    _md_cell(result.case_id),
                    _md_cell(result.session_mode),
                    str(result.metrics.tool_call_count),
                    str(result.metrics.model_retries),
                    str(result.metrics.step_retries),
                    f"{result.metrics.elapsed_seconds:.3f}s",
                    f"{result.metrics.agent_latency_seconds:.3f}s",
                    f"{result.metrics.llm_latency_seconds:.3f}s",
                    _md_cell(_fmt_cost(result.metrics.cost)),
                    _md_cell(_fmt_spend(result.metrics.estimated_spend)),
                ]
            )
            + " |"
        )
    failed = [result for result in results if result.failure_reasons]
    if failed:
        lines.extend(["", "## Failure Reasons", ""])
        for result in failed:
            lines.extend([f"### {result.case_id}", ""])
            for reason in result.failure_reasons:
                lines.append(f"- `{reason.category}` from `{reason.source}`: {reason.message}")
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def snapshot_files(root: Path) -> dict[str, str]:
    files: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(root).as_posix()
        if rel == ".mikucli" or rel.startswith(".mikucli/"):
            continue
        files[rel] = _hash_file(path)
    return files


def _approval_recorder(workspace: Path, approvals: list[ApprovalRecord]):
    workspace = workspace.resolve()

    def confirm(request: ToolApprovalRequest) -> bool:
        approved = Path(request.workspace).resolve() == workspace and _approval_details_are_local(request.details)
        approvals.append(
            ApprovalRecord(
                tool_name=request.tool_name,
                risk_level=request.risk_level.value,
                summary=request.summary,
                details=request.details,
                approved=approved,
            )
        )
        return approved

    return confirm


def _approval_details_are_local(details: str) -> bool:
    lowered = details.casefold()
    blocked = ("..", "~", "$home", "%userprofile%", "/etc/", "/home/")
    return not any(token in lowered for token in blocked)


def _hash_file(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def _safe_case_path(case_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", case_id)


def _compact(text: str, limit: int = 2000) -> str:
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "..."


def _component_spend(tokens: int | None, price_per_million: float | None) -> float | None:
    if tokens is None or price_per_million is None:
        return None
    return round(tokens * price_per_million / 1_000_000, 8)


def _read_log_events(path: Path) -> list[dict[str, Any]]:
    if not path or not path.is_file():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    events = payload.get("events")
    return events if isinstance(events, list) else []


def _fmt_cost(cost: EvalCost) -> str:
    prompt = _fmt_optional_int(cost.prompt_tokens)
    completion = _fmt_optional_int(cost.completion_tokens)
    total = _fmt_optional_int(cost.total_tokens)
    return f"prompt={prompt}, completion={completion}, total={total}"


def _fmt_spend(spend: EstimatedSpend | None) -> str:
    if spend is None:
        return "unknown"
    return f"prompt={_fmt_optional_float(spend.prompt)}, completion={_fmt_optional_float(spend.completion)}, total={_fmt_optional_float(spend.total)}"


def _fmt_percent(value: float) -> str:
    return f"{value * 100:.1f}%"


def _fmt_optional_int(value: int | None) -> str:
    return "unknown" if value is None else str(value)


def _fmt_optional_float(value: float | None) -> str:
    return "unknown" if value is None else f"{value:.8f}".rstrip("0").rstrip(".")


def _md_cell(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")

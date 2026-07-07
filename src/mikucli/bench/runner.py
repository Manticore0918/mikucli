from __future__ import annotations

import json
import re
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

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
    BenchmarkResult,
    CheckResult,
    ChatClient,
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


class BenchmarkRunner:
    def __init__(
        self,
        *,
        root: Path,
        client: ChatClient,
        model: str,
        max_steps: int = 30,
        context_window_tokens: int = 128000,
    ) -> None:
        self.root = root.resolve()
        self.client = client
        self.model = model
        self.max_steps = max_steps
        self.context_window_tokens = context_window_tokens
        self.run_id = new_session_id()
        self.bench_root = self.root / ".mikucli" / "bench"
        self.workspaces_root = self.bench_root / "workspaces" / self.run_id
        self.results_root = self.bench_root / "runs"

    def run(self, cases: Iterable[BenchmarkCase] | None = None) -> tuple[list[BenchmarkResult], Path]:
        selected = list(cases if cases is not None else all_benchmark_cases())
        if not selected:
            raise BenchmarkError("no benchmark cases selected")
        self.workspaces_root.mkdir(parents=True, exist_ok=True)
        self.results_root.mkdir(parents=True, exist_ok=True)
        results = [self.run_case(case) for case in selected]
        result_path = self._write_results(results)
        return results, result_path

    def run_case(self, case: BenchmarkCase) -> BenchmarkResult:
        workspace = self.workspaces_root / _safe_case_path(case.id)
        workspace.mkdir(parents=True, exist_ok=True)
        setup = case.task.setup(workspace)
        before_files = snapshot_files(workspace)
        approvals: list[ApprovalRecord] = []
        base_tools = self._build_tools(
            workspace=workspace,
            mode=case.session_mode,
            setup=setup,
            approvals=approvals,
        )
        tools = RecordingToolSet(base_tools)
        console = BenchmarkConsole()
        started = time.perf_counter()
        try:
            session = self._build_session(
                workspace=workspace,
                mode=case.session_mode,
                tools=tools,
                console=console,
            )
            session_result = session.run_turn(case.task.prompt)
        finally:
            tools.close()
        elapsed = time.perf_counter() - started
        after_files = snapshot_files(workspace)
        final_answer = console.answers[-1] if console.answers else ""
        if not final_answer and "session_result" in locals():
            final_answer = session_result.final_answer
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
        check_results: list[CheckResult] = []
        for check in case.task.checks:
            messages = check(context)
            check_results.append(CheckResult(name=getattr(check, "__name__", "check"), passed=not messages, messages=messages))
        passed = all(check.passed for check in check_results)
        return BenchmarkResult(
            case_id=case.id,
            task_id=case.task.id,
            session_mode=case.session_mode.value,
            passed=passed,
            check_results=check_results,
            final_answer=final_answer,
            changed_paths=context.changed_paths,
            tool_calls=tools.calls,
            approvals=approvals,
            run_log_path=str(session_result.log_path),
            workspace=str(workspace),
            model=self.model,
            elapsed_seconds=round(elapsed, 3),
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
    ) -> AgentSession | OrchestratorSession:
        kwargs = {
            "client": self.client,
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

    def _write_results(self, results: list[BenchmarkResult]) -> Path:
        path = self.results_root / f"{self.run_id}.json"
        payload = {
            "run_id": self.run_id,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "model": self.model,
            "results": [asdict(result) for result in results],
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return path


def run_benchmarks(
    *,
    root: Path,
    client: ChatClient,
    model: str,
    case_ids: set[str] | None = None,
    max_steps: int = 30,
    context_window_tokens: int = 128000,
) -> tuple[list[BenchmarkResult], Path]:
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
    ).run(cases)


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

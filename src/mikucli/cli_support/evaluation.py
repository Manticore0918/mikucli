from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal

from mikucli.config import Config
from mikucli.evaluation.bench.runner import run_benchmarks, summarize_results
from mikucli.evaluation.bench.tasks import all_benchmark_cases
from mikucli.llm import ModelClient
from mikucli.memory import MemoryRetriever


CaseStarted = Callable[[Any], None]
CaseFinished = Callable[[Any], None]
EvalResult = tuple[list[Any], Path, Path]
EvalCompleted = Callable[[EvalResult, bool], None]
EvalFailed = Callable[[Exception], None]
EvalCaseSelection = Literal["all", "single", "multi"]
EvalRunner = Callable[
    [Callable[[], bool], CaseStarted | None, CaseFinished | None, EvalCaseSelection],
    EvalResult,
]


@dataclass(frozen=True)
class EvalStatus:
    running: bool
    exclusive: bool
    started_cases: int
    completed_cases: int

    @property
    def active_cases(self) -> int:
        return max(0, self.started_cases - self.completed_cases) if self.running else 0


class EvalRunController:
    """Coordinate foreground and background eval-suite execution."""

    def __init__(self, runner: EvalRunner) -> None:
        self.runner = runner
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None
        self.result: EvalResult | None = None
        self.error: Exception | None = None
        self.lock = threading.RLock()
        self.started_cases = 0
        self.completed_cases = 0
        self.exclusive = False

    def is_running(self) -> bool:
        with self.lock:
            thread = self.thread
            return thread is not None and thread.is_alive()

    def status(self) -> EvalStatus:
        with self.lock:
            thread = self.thread
            return EvalStatus(
                running=thread is not None and thread.is_alive(),
                exclusive=self.exclusive,
                started_cases=self.started_cases,
                completed_cases=self.completed_cases,
            )

    def start(
        self,
        *,
        background: bool,
        exclusive: bool = False,
        on_case_started: CaseStarted | None = None,
        on_case_finished: CaseFinished | None = None,
        on_completed: EvalCompleted | None = None,
        on_failed: EvalFailed | None = None,
        case_selection: EvalCaseSelection = "all",
    ) -> EvalResult | None:
        with self.lock:
            if self.is_running():
                raise RuntimeError("eval suite is already running")
            self.stop_event.clear()
            self.result = None
            self.error = None
            self.started_cases = 0
            self.completed_cases = 0
            self.exclusive = exclusive
            if background:
                self.thread = threading.Thread(
                    target=self._run,
                    args=(on_case_started, on_case_finished, on_completed, on_failed, case_selection),
                    name="mikucli-eval-suite",
                    daemon=False,
                )
                self.thread.start()
                return None
        self._run(on_case_started, on_case_finished, on_completed, on_failed, case_selection)
        return self._outcome()

    def request_stop(self) -> bool:
        """Request a cooperative stop without blocking the caller."""

        with self.lock:
            thread = self.thread
            if thread is None or not thread.is_alive():
                return False
            self.stop_event.set()
            return True

    def stop(self) -> bool:
        """Request a cooperative stop without blocking the caller."""

        return self.request_stop()

    def wait(self, timeout: float | None = None) -> EvalResult | None:
        """Wait for a background run and return its outcome even if it already finished."""

        with self.lock:
            thread = self.thread
        if thread is not None:
            if thread is threading.current_thread():
                raise RuntimeError("eval suite cannot wait for itself")
            thread.join(timeout)
            if thread.is_alive():
                return None
        return self._outcome()

    def shutdown(self) -> None:
        """Stop an active run and wait until it has finalized its report."""

        self.request_stop()
        try:
            self.wait()
        except Exception:
            # Background failures are delivered through on_failed. Shutdown must
            # still finish the rest of the CLI cleanup.
            pass

    def _run(
        self,
        on_case_started: CaseStarted | None = None,
        on_case_finished: CaseFinished | None = None,
        on_completed: EvalCompleted | None = None,
        on_failed: EvalFailed | None = None,
        case_selection: EvalCaseSelection = "all",
    ) -> None:
        def case_started(case: Any) -> None:
            with self.lock:
                self.started_cases += 1
            if on_case_started is not None:
                on_case_started(case)

        def case_finished(result: Any) -> None:
            with self.lock:
                self.completed_cases += 1
            if on_case_finished is not None:
                on_case_finished(result)

        try:
            result = self.runner(self.stop_event.is_set, case_started, case_finished, case_selection)
        except Exception as exc:  # pragma: no cover - defensive capture for background threads.
            with self.lock:
                self.error = exc
            if on_failed is not None:
                on_failed(exc)
        else:
            stopped = self.stop_event.is_set()
            with self.lock:
                self.result = result
            if on_completed is not None:
                on_completed(result, stopped)

    def _outcome(self) -> EvalResult | None:
        with self.lock:
            error = self.error
            result = self.result
        if error is not None:
            raise error
        return result


def eval_runner(
    *,
    client: ModelClient,
    config: Config,
    max_steps: int,
    memory_retriever: MemoryRetriever,
) -> EvalRunner:
    def run(
        stop_requested: Callable[[], bool],
        on_case_started: CaseStarted | None = None,
        on_case_finished: CaseFinished | None = None,
        case_selection: EvalCaseSelection = "all",
    ) -> tuple[list[Any], Path, Path]:
        case_ids = _case_ids_for_selection(case_selection)
        return run_benchmarks(
            root=config.workspace,
            client=client,
            model=config.model,
            case_ids=case_ids,
            max_steps=max_steps,
            context_window_tokens=config.context_window_tokens,
            memory_retriever=memory_retriever,
            stop_requested=stop_requested,
            on_case_started=on_case_started,
            on_case_finished=on_case_finished,
        )

    return run


def _case_ids_for_selection(case_selection: EvalCaseSelection) -> set[str] | None:
    if case_selection == "all":
        return None
    if case_selection not in {"single", "multi"}:
        raise ValueError(f"unknown eval case selection: {case_selection}")
    uses_multi_agent = case_selection == "multi"
    return {
        case.id
        for case in all_benchmark_cases()
        if case.session_mode.uses_multi_agent == uses_multi_agent
    }


def print_eval_case_started(case: Any) -> None:
    print(f"mikucli: RUNNING: {case.id}")


def print_eval_case_finished(result: Any) -> None:
    status = "MISSION SUCCEED" if result.passed else "MISSION FAILED"
    metrics = result.metrics
    print(
        f"mikucli: {status}: {result.case_id} "
        f"(latency={metrics.elapsed_seconds:.3f}s, "
        f"tool_calls={metrics.tool_call_count}, "
        f"model_retries={metrics.model_retries}, "
        f"step_retries={metrics.step_retries})"
    )
    for reason in result.failure_reasons:
        print(f"mikucli:   failure [{reason.category}/{reason.source}]: {reason.message}")


def print_eval_summary(
    results: list[Any],
    result_path: Path,
    report_path: Path,
    *,
    stopped: bool,
) -> None:
    summary = summarize_results(results, stopped=stopped)
    status = "stopped" if stopped else "complete"
    print(f"mikucli: eval suite {status}: {summary.passed_cases}/{summary.total_cases} benchmark cases passed")
    print(f"mikucli: success rate: {summary.success_rate * 100:.1f}%")
    print(
        "mikucli: "
        f"tool_calls={summary.tool_call_count}, "
        f"model_retries={summary.model_retries}, "
        f"step_retries={summary.step_retries}, "
        f"latency={summary.elapsed_seconds:.3f}s"
    )
    print(f"mikucli: benchmark results: {result_path}")
    print(f"mikucli: benchmark report: {report_path}")

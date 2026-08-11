from __future__ import annotations

import threading
from pathlib import Path
from typing import Any, Callable

from mikucli.config import Config
from mikucli.evaluation.bench.runner import run_benchmarks, summarize_results
from mikucli.llm import BigModelClient
from mikucli.memory import MemoryRetriever


CaseStarted = Callable[[Any], None]
CaseFinished = Callable[[Any], None]
EvalResult = tuple[list[Any], Path, Path]
EvalCompleted = Callable[[EvalResult, bool], None]
EvalFailed = Callable[[Exception], None]
EvalRunner = Callable[[Callable[[], bool], CaseStarted | None, CaseFinished | None], EvalResult]


class EvalRunController:
    """Coordinate foreground and background eval-suite execution."""

    def __init__(self, runner: EvalRunner) -> None:
        self.runner = runner
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None
        self.result: EvalResult | None = None
        self.error: Exception | None = None
        self.lock = threading.RLock()

    def is_running(self) -> bool:
        with self.lock:
            thread = self.thread
            return thread is not None and thread.is_alive()

    def start(
        self,
        *,
        background: bool,
        on_case_started: CaseStarted | None = None,
        on_case_finished: CaseFinished | None = None,
        on_completed: EvalCompleted | None = None,
        on_failed: EvalFailed | None = None,
    ) -> EvalResult | None:
        with self.lock:
            if self.is_running():
                raise RuntimeError("eval suite is already running")
            self.stop_event.clear()
            self.result = None
            self.error = None
            if background:
                self.thread = threading.Thread(
                    target=self._run,
                    args=(on_case_started, on_case_finished, on_completed, on_failed),
                    name="mikucli-eval-suite",
                    daemon=False,
                )
                self.thread.start()
                return None
        self._run(on_case_started, on_case_finished, on_completed, on_failed)
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
    ) -> None:
        try:
            result = self.runner(self.stop_event.is_set, on_case_started, on_case_finished)
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
    client: BigModelClient,
    config: Config,
    max_steps: int,
    memory_retriever: MemoryRetriever,
) -> EvalRunner:
    def run(
        stop_requested: Callable[[], bool],
        on_case_started: CaseStarted | None = None,
        on_case_finished: CaseFinished | None = None,
    ) -> tuple[list[Any], Path, Path]:
        return run_benchmarks(
            root=config.workspace,
            client=client,
            model=config.model,
            max_steps=max_steps,
            context_window_tokens=config.context_window_tokens,
            memory_retriever=memory_retriever,
            stop_requested=stop_requested,
            on_case_started=on_case_started,
            on_case_finished=on_case_finished,
        )

    return run


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

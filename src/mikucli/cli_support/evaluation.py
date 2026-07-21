from __future__ import annotations

import threading
from pathlib import Path
from typing import Any, Callable

from mikucli.config import Config
from mikucli.evaluation.bench.runner import run_benchmarks, summarize_results
from mikucli.llm import BigModelClient


CaseStarted = Callable[[Any], None]
CaseFinished = Callable[[Any], None]
EvalRunner = Callable[[Callable[[], bool], CaseStarted | None, CaseFinished | None], tuple[list[Any], Path, Path]]


class EvalRunController:
    """Coordinate foreground and background eval-suite execution."""

    def __init__(self, runner: EvalRunner) -> None:
        self.runner = runner
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None
        self.result: tuple[list[Any], Path, Path] | None = None
        self.error: Exception | None = None
        self.lock = threading.Lock()

    def is_running(self) -> bool:
        thread = self.thread
        return thread is not None and thread.is_alive()

    def start(
        self,
        *,
        background: bool,
        on_case_started: CaseStarted | None = None,
        on_case_finished: CaseFinished | None = None,
    ) -> tuple[list[Any], Path, Path] | None:
        with self.lock:
            if self.is_running():
                raise RuntimeError("eval suite is already running")
            self.stop_event.clear()
            self.result = None
            self.error = None
            if background:
                self.thread = threading.Thread(
                    target=self._run,
                    args=(on_case_started, on_case_finished),
                    name="mikucli-eval-suite",
                    daemon=True,
                )
                self.thread.start()
                return None
        if background:
            return None
        self._run(on_case_started, on_case_finished)
        if self.error is not None:
            raise self.error
        return self.result

    def stop(self) -> tuple[list[Any], Path, Path] | None:
        if not self.is_running():
            return None
        self.stop_event.set()
        thread = self.thread
        if thread is not None:
            thread.join()
        if self.error is not None:
            raise self.error
        return self.result

    def _run(
        self,
        on_case_started: CaseStarted | None = None,
        on_case_finished: CaseFinished | None = None,
    ) -> None:
        try:
            self.result = self.runner(self.stop_event.is_set, on_case_started, on_case_finished)
        except Exception as exc:  # pragma: no cover - defensive capture for background threads.
            self.error = exc


def eval_runner(*, client: BigModelClient, config: Config, max_steps: int) -> EvalRunner:
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
        f"(total={metrics.elapsed_seconds:.3f}s, "
        f"agent={metrics.agent_latency_seconds:.3f}s, "
        f"llm={metrics.llm_latency_seconds:.3f}s, "
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
        f"total_latency={summary.elapsed_seconds:.3f}s, "
        f"agent_latency={summary.agent_latency_seconds:.3f}s, "
        f"llm_latency={summary.llm_latency_seconds:.3f}s"
    )
    print(f"mikucli: benchmark results: {result_path}")
    print(f"mikucli: benchmark report: {report_path}")

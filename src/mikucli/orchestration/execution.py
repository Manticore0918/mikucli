from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from mikucli.logs import RunLog
from mikucli.react import AgentSession

from .models import ExecutionStep, ReviewDecision
from .parsing import parse_review_decision


class StepExecutor:
    """Execute dependency-ready steps and coordinate worker review cycles."""

    def __init__(self, session: Any) -> None:
        self.session = session

    def execute_steps(
        self,
        task_prompt: str,
        steps: list[ExecutionStep],
        run_log: RunLog,
        *,
        trace_id: str = "",
        workflow_span_id: str = "",
    ) -> None:
        step_by_id = {step.id: step for step in steps}
        self.session._current_step_by_id = step_by_id
        remaining = {step.id for step in steps}
        worker_index = 0
        while remaining:
            skipped = self._skip_blocked_steps(step_by_id, remaining)
            for step in skipped:
                run_log.add_event("step_skipped", step_id=step.id, reason=step.skipped_reason)
            remaining.difference_update(step.id for step in skipped)

            ready = [
                step_by_id[step_id]
                for step_id in sorted(remaining)
                if all(step_by_id[dependency].status == "passed" for dependency in step_by_id[step_id].depends_on)
            ]
            if not ready:
                for step_id in sorted(remaining):
                    step = step_by_id[step_id]
                    step.status = "skipped"
                    step.skipped_reason = "No executable dependency batch remained."
                    run_log.add_event("step_skipped", step_id=step.id, reason=step.skipped_reason)
                break

            workers = self.session._workers()
            for start in range(0, len(ready), len(workers)):
                batch = ready[start : start + len(workers)]
                assignments: list[tuple[ExecutionStep, AgentSession, str]] = []
                for step in batch:
                    worker_id, worker = workers[worker_index % len(workers)]
                    worker_index += 1
                    step.assigned_worker = worker_id
                    assignments.append((step, worker, worker_id))

                with ThreadPoolExecutor(max_workers=len(assignments)) as executor:
                    futures = {
                        executor.submit(
                            self._run_step_with_review,
                            task_prompt,
                            step,
                            worker,
                            worker_id,
                            run_log,
                            trace_id,
                            workflow_span_id,
                        ): step
                        for step, worker, worker_id in assignments
                    }
                    for future in as_completed(futures):
                        step = futures[future]
                        try:
                            future.result()
                        except Exception as exc:  # pragma: no cover - defensive worker-thread guard.
                            step.status = "failed"
                            step.feedback = f"Step execution raised {type(exc).__name__}: {exc}"
                            run_log.add_event("step_failed", step_id=step.id, error=step.feedback)
            remaining.difference_update(step.id for step in ready)

    def dependency_context(self, step: ExecutionStep) -> str:
        if not step.depends_on:
            return ""
        lines: list[str] = []
        for dependency_id in step.depends_on:
            dependency = self.session._current_step_by_id.get(dependency_id)
            if dependency is None or dependency.status != "passed":
                continue
            lines.append(
                f"{dependency.id}: {dependency.title or dependency.task}\n"
                f"Result: {dependency.result}"
            )
        return "\n\n".join(lines)[:500]

    def _run_step_with_review(
        self,
        task_prompt: str,
        step: ExecutionStep,
        worker: AgentSession,
        worker_id: str,
        run_log: RunLog,
        trace_id: str = "",
        workflow_span_id: str = "",
    ) -> None:
        session = self.session
        step_span_id = session.trace_recorder.start_span(
            trace_id=trace_id,
            name="orchestrator.step",
            kind="orchestrator",
            parent_span_id=workflow_span_id,
            attributes={
                "step.id": step.id,
                "step.title": step.title,
                "step.depends_on": step.depends_on,
                "worker.id": worker_id,
            },
        )
        reviewer = session._reviewer()
        reviewer_id = session._subagent_id_by_role("reviewer")
        feedback = ""
        status = "ok"
        try:
            for attempt in range(1, session.max_step_attempts + 1):
                worker_result = self._run_worker_step(
                    task_prompt=task_prompt,
                    step=step,
                    dependency_context=self.dependency_context(step),
                    worker=worker,
                    worker_id=worker_id,
                    attempt=attempt,
                    feedback=feedback,
                    run_log=run_log,
                    trace_id=trace_id,
                    step_span_id=step_span_id,
                )
                decision = self._review_step_result(
                    task_prompt=task_prompt,
                    step=step,
                    worker_result=worker_result,
                    reviewer=reviewer,
                    run_log=run_log,
                    attempt=attempt,
                    trace_id=trace_id,
                    step_span_id=step_span_id,
                )
                session.trace_recorder.add_event(
                    step_span_id,
                    "review.completed",
                    {
                        "step.id": step.id,
                        "step.attempt": attempt,
                        "review.approved": decision.approved,
                        "review.issue_count": len(decision.issues),
                    },
                )
                run_log.add_event(
                    "step_reviewed",
                    step_id=step.id,
                    attempt=attempt,
                    approved=decision.approved,
                    problems=decision.problems,
                    summary=decision.summary,
                )
                if decision.approved:
                    step.status = "passed"
                    return
                feedback = decision.problems or "Reviewer rejected the step without specific problems."
            step.status = "failed"
            step.feedback = feedback
            status = "error"
        except BaseException as exc:
            status = "error"
            session.trace_recorder.add_event(
                step_span_id,
                "step.error",
                {"step.id": step.id, "error.type": type(exc).__name__, "error.message": str(exc)},
            )
            raise
        finally:
            worker.clear_chat_history()
            run_log.add_event(
                "subagent_chat_history_cleared",
                step_id=step.id,
                worker=worker_id,
                reviewer=reviewer_id,
            )
            session.trace_recorder.end_span(
                step_span_id,
                status=status,
                attributes={
                    "step.id": step.id,
                    "step.status": step.status,
                    "step.attempts": step.attempts,
                    "review.issue_count": len(step.feedback.splitlines()) if step.feedback else 0,
                },
            )

    def _run_worker_step(
        self,
        *,
        task_prompt: str,
        step: ExecutionStep,
        dependency_context: str,
        worker: AgentSession,
        worker_id: str,
        attempt: int,
        feedback: str,
        run_log: RunLog,
        trace_id: str = "",
        step_span_id: str = "",
    ) -> str:
        session = self.session
        step.status = "running"
        step.attempts = attempt
        session.console.progress(f"{worker_id} executing [{step.id}]: {step.title or step.task}")
        worker_result = worker.run_turn(
            session._worker_task(task_prompt, step, attempt, feedback, dependency_context),
            trace_id=trace_id,
            parent_span_id=step_span_id,
            span_name="subagent.turn",
            span_kind="subagent",
            span_attributes={
                "subagent.id": worker_id,
                "subagent.role": "worker",
                "step.id": step.id,
                "step.attempt": attempt,
            },
            session_mode="multi_agent",
        )
        step.result = worker_result.final_answer
        run_log.add_event(
            "step_worker_result",
            step_id=step.id,
            worker=worker_id,
            attempt=attempt,
            content=worker_result.final_answer,
        )
        return worker_result.final_answer

    def _review_step_result(
        self,
        *,
        task_prompt: str,
        step: ExecutionStep,
        worker_result: str,
        reviewer: AgentSession,
        run_log: RunLog,
        attempt: int,
        trace_id: str = "",
        step_span_id: str = "",
    ) -> ReviewDecision:
        session = self.session
        session.console.progress(f"reviewer reviewing the results of [{step.id}]")
        with session._reviewer_lock:
            try:
                review_result = reviewer.run_turn(
                    session._review_task(task_prompt, step, worker_result),
                    trace_id=trace_id,
                    parent_span_id=step_span_id,
                    span_name="subagent.turn",
                    span_kind="subagent",
                    span_attributes={
                        "subagent.id": session._subagent_id_by_role("reviewer"),
                        "subagent.role": "reviewer",
                        "step.id": step.id,
                        "step.attempt": attempt,
                    },
                    session_mode="multi_agent",
                )
            finally:
                reviewer.clear_chat_history()
        decision = parse_review_decision(review_result.final_answer)
        step.review_summary = decision.summary
        step.feedback = decision.problems
        run_log.add_event(
            "step_review_result",
            step_id=step.id,
            attempt=attempt,
            content=review_result.final_answer,
        )
        if decision.approved:
            session.console.progress(f"[{step.id}] review approved")
        else:
            session.console.progress(f"[{step.id}] review rejected: {decision.problems}")
        return decision

    @staticmethod
    def _skip_blocked_steps(
        step_by_id: dict[str, ExecutionStep],
        remaining: set[str],
    ) -> list[ExecutionStep]:
        skipped: list[ExecutionStep] = []
        for step_id in sorted(remaining):
            step = step_by_id[step_id]
            blockers = [
                dependency
                for dependency in step.depends_on
                if step_by_id[dependency].status in {"failed", "skipped"}
            ]
            if blockers:
                step.status = "skipped"
                step.skipped_reason = "Blocked by failed or skipped dependencies: " + ", ".join(blockers)
                skipped.append(step)
        return skipped

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from mikucli.agent_runtime.cancellation import raise_if_stop_requested
from mikucli.logs import RunLog
from mikucli.skills import Skill

from .models import ExecutionStep
from .parsing import parse_execution_plan

if TYPE_CHECKING:
    from mikucli.multi_agent import OrchestratorSession


@dataclass(frozen=True)
class PlanningResult:
    steps: list[ExecutionStep] | None
    direct_answer: str = ""


class PlanCoordinator:
    """Translate planner output, with one repair pass for malformed plan-shaped JSON."""

    def __init__(self, session: OrchestratorSession) -> None:
        self.session = session

    def create_plan(
        self,
        task_prompt: str,
        run_log: RunLog,
        *,
        active_skill: Skill | None = None,
        trace_id: str = "",
        workflow_span_id: str = "",
        stop_requested: Callable[[], bool] | None = None,
    ) -> PlanningResult:
        session = self.session
        raise_if_stop_requested(stop_requested)
        session.console.progress("phase 1: planning")
        plan_span_id = session.trace_recorder.start_span(
            trace_id=trace_id,
            name="orchestrator.plan",
            kind="orchestrator",
            parent_span_id=workflow_span_id,
        )
        planner_answer = ""
        planner_attempts = 1
        span_ended = False
        try:
            planner = session._planner()
            planner_result = planner.run_turn(
                _planner_task(task_prompt),
                active_skill=active_skill,
                trace_id=trace_id,
                parent_span_id=plan_span_id,
                span_name="subagent.turn",
                span_kind="subagent",
                span_attributes=self._span_attributes(planner_attempts),
                session_mode="multi_agent",
                stop_requested=stop_requested,
            )
            raise_if_stop_requested(stop_requested)
            planner_answer = planner_result.final_answer
            run_log.add_event("planner_result", attempt=planner_attempts, content=planner_answer)
            try:
                steps = parse_execution_plan(planner_answer)
            except ValueError as exc:
                if not _looks_like_execution_plan(planner_answer):
                    direct_answer = _direct_answer(planner_answer, exc, run_log)
                    return PlanningResult(steps=None, direct_answer=direct_answer)
                run_log.add_event("planner_plan_rejected", attempt=planner_attempts, error=str(exc))
                session.console.progress("planner repairing invalid plan")
                planner_attempts += 1
                planner_result = planner.run_turn(
                    _planner_repair_task(str(exc)),
                    active_skill=active_skill,
                    trace_id=trace_id,
                    parent_span_id=plan_span_id,
                    span_name="subagent.turn",
                    span_kind="subagent",
                    span_attributes=self._span_attributes(planner_attempts),
                    session_mode="multi_agent",
                    stop_requested=stop_requested,
                )
                raise_if_stop_requested(stop_requested)
                planner_answer = planner_result.final_answer
                run_log.add_event("planner_result", attempt=planner_attempts, content=planner_answer)
                try:
                    steps = parse_execution_plan(planner_answer)
                except ValueError as repair_exc:
                    if not _looks_like_execution_plan(planner_answer):
                        direct_answer = _direct_answer(planner_answer, repair_exc, run_log)
                        return PlanningResult(steps=None, direct_answer=direct_answer)
                    raise ValueError(
                        f"planner returned an invalid execution plan after repair: {repair_exc}"
                    ) from repair_exc
            session.trace_recorder.add_event(
                plan_span_id,
                "plan.translated",
                {"orchestrator.step_count": len(steps)},
            )
            run_log.add_event(
                "execution_plan_translated",
                steps=[
                    {"id": step.id, "title": step.title, "depends_on": step.depends_on}
                    for step in steps
                ],
            )
            return PlanningResult(steps=steps)
        except BaseException as exc:
            session.trace_recorder.end_span(
                plan_span_id,
                status="error",
                attributes={"error.type": type(exc).__name__, "error.message": str(exc)},
            )
            span_ended = True
            raise
        finally:
            if not span_ended:
                session.trace_recorder.end_span(
                    plan_span_id,
                    attributes={
                        "planner.answer.length": len(planner_answer),
                        "planner.attempts": planner_attempts,
                    },
                )

    def _span_attributes(self, attempt: int) -> dict[str, Any]:
        session = self.session
        return {
            "subagent.id": session._subagent_id_by_role("planner"),
            "subagent.role": "planner",
            "plan.attempt": attempt,
        }


def _planner_task(task_prompt: str) -> str:
    return (
        "Create the execution plan for this task. Return only the required JSON object.\n\n"
        f"Task:\n{task_prompt}"
    )


def _planner_repair_task(parse_error: str) -> str:
    return (
        "Your previous response was shaped like an execution plan but was invalid. "
        f"Fix this parser error: {parse_error}. Return a complete replacement JSON object "
        "using the required steps schema. Do not use placeholders, markdown, or prose."
    )


def _looks_like_execution_plan(raw: str) -> bool:
    cleaned = raw.strip()
    if "{" not in cleaned:
        return False
    return re.search(r"(?:[\"']steps[\"']|\bsteps\b)\s*:", cleaned, re.IGNORECASE) is not None


def _direct_answer(raw: str, parse_error: ValueError, run_log: RunLog) -> str:
    answer = raw.strip()
    if not answer:
        raise parse_error
    run_log.add_event("planner_direct_answer", content=answer, parse_error=str(parse_error))
    return answer

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
import json
from pathlib import Path
import re
from threading import Lock
from typing import Any, Literal

from .llm import BigModelClient, TokenUsage
from .logs import RunLog, RunLogWriter, new_session_id
from .memory import LongTermMemory, SessionMemory
from .observability import TraceRecorder, create_trace_recorder
from .react import BASE_AGENT_INSTRUCTIONS, AgentSession, Console, SessionResult, ToolSet
from .tools import ToolResult


StepStatus = Literal["pending", "running", "passed", "failed", "skipped"]


READ_ONLY_AGENT_INSTRUCTIONS = """You may use only the tools made available to you. Those tools are read-only inspection tools. Do not write files, run shell commands, save memory, or perform external mutation.
Return the requested JSON or concise answer from the information the orchestrator gives you and any read-only inspection you perform.
Do not reveal raw internal reasoning or chain-of-thought.
"""


@dataclass(frozen=True)
class SubAgentSpec:
    id: str
    role: str
    purpose: str


@dataclass
class ExecutionStep:
    id: str
    task: str
    title: str = ""
    depends_on: list[str] = field(default_factory=list)
    status: StepStatus = "pending"
    assigned_worker: str = ""
    attempts: int = 0
    result: str = ""
    review_summary: str = ""
    feedback: str = ""
    skipped_reason: str = ""


@dataclass(frozen=True)
class ReviewDecision:
    approved: bool
    summary: str = ""
    issues: list[str] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return self.approved

    @property
    def feedback(self) -> str:
        return self.problems

    @property
    def problems(self) -> str:
        parts: list[str] = []
        if self.issues:
            parts.append("Issues: " + "; ".join(self.issues))
        if self.suggestions:
            parts.append("Suggestions: " + "; ".join(self.suggestions))
        return "\n".join(parts)


DEFAULT_SUBAGENTS: tuple[SubAgentSpec, ...] = (
    SubAgentSpec(
        id="planner-1",
        role="planner",
        purpose="Break down the task, identify dependencies, and produce an execution plan.",
    ),
    SubAgentSpec(
        id="worker-1",
        role="worker",
        purpose="Execute implementation work and gather concrete workspace evidence.",
    ),
    SubAgentSpec(
        id="worker-2",
        role="worker",
        purpose="Execute implementation work and gather concrete workspace evidence.",
    ),
    SubAgentSpec(
        id="reviewer-1",
        role="reviewer",
        purpose="Review completed steps for defects, missed requirements, and verification gaps.",
    ),
)


def orchestrator_system_prompt(subagents: tuple[SubAgentSpec, ...] = DEFAULT_SUBAGENTS) -> str:
    roster = "\n".join(f"- {agent.id} ({agent.role}): {agent.purpose}" for agent in subagents)
    return f"""You are mikucli's orchestrator, the main agent for the user's session.

Coordinate this Orchestrator-SubAgent team:
{roster}

Workflow:
1. Ask the planner for a JSON execution plan.
2. Translate that plan into ExecutionStep objects with dependency relations.
3. Run dependency-ready steps through workers. Steps in the same dependency batch may run simultaneously.
4. Ask the reviewer to review each completed step. Retry rejected steps with reviewer feedback.
5. Skip steps blocked by failed or skipped dependencies.
6. Summarize every step status and result into session memory before answering the user.

{BASE_AGENT_INSTRUCTIONS}
"""


def subagent_system_prompt(spec: SubAgentSpec) -> str:
    if spec.role == "planner":
        return f"""You are {spec.id}, the planner subagent in mikucli's Orchestrator-SubAgent team.

Return only a JSON object with this shape:
{{
  "steps": [
    {{
      "id": "step-1",
      "title": "Short title",
      "task": "Concrete task for a worker",
      "depends_on": []
    }}
  ]
}}

Use stable step ids. Put prerequisite step ids in "depends_on". Do not include markdown fences or explanatory prose.

{READ_ONLY_AGENT_INSTRUCTIONS}
"""
    if spec.role == "reviewer":
        return f"""You are {spec.id}, the reviewer subagent in mikucli's Orchestrator-SubAgent team.

Review only the completed step the orchestrator gives you. Return only a JSON object with this shape:
{{
  "approved": true,
  "summary": "Concise review summary",
  "issues": ["issue 1", "issue 2"],
  "suggestions": ["suggestion 1", "suggestion 2"]
}}

Set "approved" to false when the worker must rerun the step. Put concrete defects in "issues" and actionable fixes in "suggestions".

{READ_ONLY_AGENT_INSTRUCTIONS}
"""
    return f"""You are {spec.id}, a worker subagent in mikucli's Orchestrator-SubAgent team.

Execute only the step delegated by the orchestrator. Return a concise result that includes concrete findings, changed files when relevant, verification performed, and blockers.

{BASE_AGENT_INSTRUCTIONS}
"""


class OrchestratorSession:
    def __init__(
        self,
        *,
        client: BigModelClient,
        model: str,
        workspace: Path,
        tools: ToolSet,
        console: Console,
        max_steps: int = 30,
        context_window_tokens: int = 128000,
        memory_window_entries: int = 40,
        compression_threshold: float = 0.8,
        long_term_memory: LongTermMemory | None = None,
        retain_recent_rounds: int = 3,
        subagents: tuple[SubAgentSpec, ...] = DEFAULT_SUBAGENTS,
        max_step_attempts: int = 2,
        trace_recorder: TraceRecorder | None = None,
    ) -> None:
        if max_step_attempts <= 0:
            raise ValueError("max_step_attempts must be positive")
        self.model = model
        self.workspace = workspace
        self.console = console
        self.subagent_specs = subagents
        self.max_step_attempts = max_step_attempts
        self.trace_recorder = trace_recorder or create_trace_recorder(workspace)
        self.log_writer = RunLogWriter(workspace)
        self._reviewer_lock = Lock()
        self._current_step_by_id: dict[str, ExecutionStep] = {}
        self.memory = SessionMemory(
            system_message={"role": "system", "content": orchestrator_system_prompt(subagents)},
            max_active_entries=memory_window_entries,
            long_term_memory=long_term_memory,
        )
        self.subagents = {
            spec.id: AgentSession(
                client=client,
                model=model,
                workspace=workspace,
                tools=tools if spec.role == "worker" else ReadOnlyTools(tools),
                console=PrefixedConsole(console, spec.id),
                max_steps=max_steps,
                context_window_tokens=context_window_tokens,
                memory_window_entries=memory_window_entries,
                compression_threshold=compression_threshold,
                long_term_memory=long_term_memory,
                retain_recent_rounds=retain_recent_rounds,
                system_prompt=subagent_system_prompt(spec),
                agent_name=spec.id,
                trace_recorder=self.trace_recorder,
            )
            for spec in subagents
        }

    def run_turn(self, task_prompt: str) -> SessionResult:
        run_log = RunLog(
            session_id=new_session_id(),
            task_prompt=task_prompt,
            model=self.model,
            workspace=str(self.workspace),
        )
        trace_id = self.trace_recorder.start_trace(
            run_id=run_log.session_id,
            task_prompt=task_prompt,
            workspace=str(self.workspace),
            model=self.model,
            session_mode="multi_agent",
            attributes={"agent.name": "orchestrator"},
        )
        if trace_id:
            run_log.metadata["trace_id"] = trace_id
        agent_span_id = self.trace_recorder.start_span(
            trace_id=trace_id,
            name="agent.session",
            kind="agent",
            attributes={
                "agent.name": "orchestrator",
                "model": self.model,
                "workspace": str(self.workspace),
            },
        )
        workflow_span_id = self.trace_recorder.start_span(
            trace_id=trace_id,
            name="orchestrator.workflow",
            kind="orchestrator",
            parent_span_id=agent_span_id,
            attributes={"agent.name": "orchestrator"},
        )
        run_log.add_event("agent_started", agent="orchestrator")
        self.memory.add_conversation({"role": "user", "content": task_prompt}, content=task_prompt)

        final_answer = ""
        trace_status = "ok"
        try:
            self.console.progress("phase 1: planning")
            plan_span_id = self.trace_recorder.start_span(
                trace_id=trace_id,
                name="orchestrator.plan",
                kind="orchestrator",
                parent_span_id=workflow_span_id,
            )
            try:
                planner_result = self._planner().run_turn(
                    self._planner_task(task_prompt),
                    trace_id=trace_id,
                    parent_span_id=plan_span_id,
                    span_name="subagent.turn",
                    span_kind="subagent",
                    span_attributes={"subagent.id": self._subagent_id_by_role("planner"), "subagent.role": "planner"},
                    session_mode="multi_agent",
                )
                self.trace_recorder.end_span(plan_span_id, attributes={"planner.answer.length": len(planner_result.final_answer)})
                run_log.add_event("planner_result", content=planner_result.final_answer)
                try:
                    steps = parse_execution_plan(planner_result.final_answer)
                except ValueError as exc:
                    final_answer = planner_result.final_answer.strip()
                    if not final_answer:
                        raise
                    run_log.add_event("planner_direct_answer", content=final_answer, parse_error=str(exc))
                else:
                    self.trace_recorder.add_event(
                        plan_span_id,
                        "plan.translated",
                        {"orchestrator.step_count": len(steps)},
                    )
                    run_log.add_event(
                        "execution_plan_translated",
                        steps=[{"id": step.id, "title": step.title, "depends_on": step.depends_on} for step in steps],
                    )
                    self._show_plan(steps)
                    self.console.progress("phase 2: executing")
                    self._execute_steps(task_prompt, steps, run_log, trace_id=trace_id, workflow_span_id=workflow_span_id)
                    final_answer = summarize_execution(steps)
            except Exception as exc:
                self.trace_recorder.end_span(
                    plan_span_id,
                    status="error",
                    attributes={"error.type": type(exc).__name__, "error.message": str(exc)},
                )
                raise
        except ValueError as exc:
            trace_status = "error"
            final_answer = f"Could not execute the orchestrator workflow: {exc}"
            run_log.add_event("workflow_failed", error=str(exc))

        self.memory.add_fact(final_answer)
        self.memory.add_conversation({"role": "assistant", "content": final_answer}, content=final_answer)
        self.console.answer(final_answer)
        run_log.final_answer = final_answer
        self.trace_recorder.end_span(
            workflow_span_id,
            status=trace_status,
            attributes={"final_answer.length": len(final_answer)},
        )
        self.trace_recorder.end_span(
            agent_span_id,
            status=trace_status,
            attributes={"final_answer.length": len(final_answer)},
        )
        self.trace_recorder.end_trace(
            trace_id,
            status=trace_status,
            attributes={"final_answer.length": len(final_answer)},
        )
        log_path = self.log_writer.write(run_log)
        return SessionResult(final_answer=final_answer, log_path=log_path)

    def _execute_steps(
        self,
        task_prompt: str,
        steps: list[ExecutionStep],
        run_log: RunLog,
        *,
        trace_id: str = "",
        workflow_span_id: str = "",
    ) -> None:
        step_by_id = {step.id: step for step in steps}
        self._current_step_by_id = step_by_id
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

            workers = self._workers()
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
                        except Exception as exc:  # pragma: no cover - defensive guard around worker threads.
                            step.status = "failed"
                            step.feedback = f"Step execution raised {type(exc).__name__}: {exc}"
                            run_log.add_event("step_failed", step_id=step.id, error=step.feedback)
            remaining.difference_update(step.id for step in ready)

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
        step_span_id = self.trace_recorder.start_span(
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
        reviewer = self._reviewer()
        reviewer_id = self._subagent_id_by_role("reviewer")
        feedback = ""
        status = "ok"
        try:
            for attempt in range(1, self.max_step_attempts + 1):
                worker_result = self._run_worker_step(
                    task_prompt=task_prompt,
                    step=step,
                    dependency_context=self._dependency_context(step),
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
                self.trace_recorder.add_event(
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
        except Exception as exc:
            status = "error"
            self.trace_recorder.add_event(
                step_span_id,
                "step.error",
                {"step.id": step.id, "error.type": type(exc).__name__, "error.message": str(exc)},
            )
            raise
        finally:
            worker.clear_chat_history()
            run_log.add_event("subagent_chat_history_cleared", step_id=step.id, worker=worker_id, reviewer=reviewer_id)
            self.trace_recorder.end_span(
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
        step.status = "running"
        step.attempts = attempt
        self.console.progress(f"{worker_id} executing [{step.id}]: {step.title or step.task}")
        worker_result = worker.run_turn(
            self._worker_task(task_prompt, step, attempt, feedback, dependency_context),
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
        self.console.progress(f"reviewer reviewing the results of [{step.id}]")
        with self._reviewer_lock:
            try:
                review_result = reviewer.run_turn(
                    self._review_task(task_prompt, step, worker_result),
                    trace_id=trace_id,
                    parent_span_id=step_span_id,
                    span_name="subagent.turn",
                    span_kind="subagent",
                    span_attributes={
                        "subagent.id": self._subagent_id_by_role("reviewer"),
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
            self.console.progress(f"[{step.id}] review approved")
        else:
            self.console.progress(f"[{step.id}] review rejected: {decision.problems}")
        return decision

    def _show_plan(self, steps: list[ExecutionStep]) -> None:
        self.console.progress("plan:")
        for step in steps:
            self.console.progress(f"{step.id}: {step.title or step.task}")

    def _skip_blocked_steps(
        self,
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

    def _planner(self) -> AgentSession:
        return self._subagent_by_role("planner")

    def _reviewer(self) -> AgentSession:
        return self._subagent_by_role("reviewer")

    def _workers(self) -> list[tuple[str, AgentSession]]:
        workers = [(spec.id, self.subagents[spec.id]) for spec in self.subagent_specs if spec.role == "worker"]
        if not workers:
            raise ValueError("no worker subagents are initialized")
        return workers

    def _subagent_by_role(self, role: str) -> AgentSession:
        for spec in self.subagent_specs:
            if spec.role == role:
                return self.subagents[spec.id]
        raise ValueError(f"no {role} subagent is initialized")

    def _subagent_id_by_role(self, role: str) -> str:
        for spec in self.subagent_specs:
            if spec.role == role:
                return spec.id
        raise ValueError(f"no {role} subagent is initialized")

    def _planner_task(self, task_prompt: str) -> str:
        return (
            "Create the execution plan for this task. Return only the required JSON object.\n\n"
            f"Task:\n{task_prompt}"
        )

    def _worker_task(
        self,
        task_prompt: str,
        step: ExecutionStep,
        attempt: int,
        feedback: str,
        dependency_context: str = "",
    ) -> str:
        feedback_block = f"\nReviewer feedback to address:\n{feedback}\n" if feedback else ""
        dependency_block = (
            f"\nCompleted dependency step context (first 500 characters):\n{dependency_context}\n"
            if dependency_context
            else ""
        )
        return (
            f"Original task:\n{task_prompt}\n\n"
            f"ExecutionStep {step.id}: {step.title or step.id}\n"
            f"{step.task}\n\n"
            f"Attempt: {attempt} of {self.max_step_attempts}."
            f"{dependency_block}"
            f"{feedback_block}"
        )

    def _dependency_context(self, step: ExecutionStep) -> str:
        if not step.depends_on:
            return ""
        lines: list[str] = []
        for dependency_id in step.depends_on:
            dependency = self._current_step_by_id.get(dependency_id)
            if dependency is None or dependency.status != "passed":
                continue
            lines.append(
                f"{dependency.id}: {dependency.title or dependency.task}\n"
                f"Result: {dependency.result}"
            )
        return "\n\n".join(lines)[:500]

    def _review_task(self, task_prompt: str, step: ExecutionStep, worker_result: str) -> str:
        return (
            "Review this completed execution step. Return only the required JSON review object.\n\n"
            f"Original task:\n{task_prompt}\n\n"
            f"ExecutionStep {step.id}: {step.title or step.id}\n"
            f"{step.task}\n\n"
            f"Worker result:\n{worker_result}"
        )


def parse_execution_plan(raw: str) -> list[ExecutionStep]:
    plan = _extract_json_object(raw)
    raw_steps = plan.get("steps")
    if not isinstance(raw_steps, list) or not raw_steps:
        raise ValueError("planner output must contain a non-empty steps array")

    steps: list[ExecutionStep] = []
    seen_ids: set[str] = set()
    for index, raw_step in enumerate(raw_steps, start=1):
        if not isinstance(raw_step, dict):
            raise ValueError(f"step {index} must be an object")
        step_id = str(raw_step.get("id") or f"step-{index}").strip()
        if not step_id:
            raise ValueError(f"step {index} has an empty id")
        if step_id in seen_ids:
            raise ValueError(f"duplicate step id: {step_id}")
        task = str(raw_step.get("task") or raw_step.get("description") or "").strip()
        if not task:
            raise ValueError(f"{step_id} must include a task")
        depends_on = _string_list(raw_step.get("depends_on", raw_step.get("dependencies", raw_step.get("dependsOn", []))))
        steps.append(
            ExecutionStep(
                id=step_id,
                title=str(raw_step.get("title") or "").strip(),
                task=task,
                depends_on=depends_on,
            )
        )
        seen_ids.add(step_id)

    known_ids = {step.id for step in steps}
    for step in steps:
        unknown = [dependency for dependency in step.depends_on if dependency not in known_ids]
        if unknown:
            raise ValueError(f"{step.id} depends on unknown step id(s): {', '.join(unknown)}")
    _validate_acyclic(steps)
    return steps


def parse_review_decision(raw: str) -> ReviewDecision:
    try:
        review = _extract_json_object(raw)
    except ValueError:
        return ReviewDecision(
            approved=False,
            summary=raw.strip(),
            issues=["Reviewer did not return valid JSON."],
        )
    approved_value = review.get("approved", review.get("passed"))
    issues = _string_list(review.get("issues", []))
    suggestions = _string_list(review.get("suggestions", []))
    legacy_problem = str(review.get("problems", review.get("feedback", "")) or "").strip()
    if legacy_problem and not issues:
        issues = [legacy_problem]
    return ReviewDecision(
        approved=_bool_value(approved_value),
        summary=str(review.get("summary") or "").strip(),
        issues=issues,
        suggestions=suggestions,
    )


def summarize_execution(steps: list[ExecutionStep]) -> str:
    lines = ["Execution summary:"]
    for step in steps:
        detail = step.result.strip() if step.status == "passed" else step.feedback.strip()
        if step.status == "skipped":
            detail = step.skipped_reason
        if not detail:
            detail = "No additional detail."
        lines.append(
            f"- {step.id} [{step.status}]"
            f"{f' via {step.assigned_worker}' if step.assigned_worker else ''}: {step.title or step.task}"
        )
        lines.append(f"  Result: {_compact(detail)}")
        if step.review_summary:
            lines.append(f"  Review: {_compact(step.review_summary)}")
    skipped = [step.id for step in steps if step.status == "skipped"]
    if skipped:
        lines.append("Skipped steps: " + ", ".join(skipped))
    return "\n".join(lines)


def _extract_json_object(raw: str) -> dict[str, Any]:
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("expected a JSON object")
    try:
        parsed = json.loads(cleaned[start : end + 1])
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON object: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ValueError("expected a JSON object")
    return parsed


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value.strip() else []
    if not isinstance(value, list):
        raise ValueError("dependencies must be a list of step ids")
    return [str(item).strip() for item in value if str(item).strip()]


def _bool_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().casefold() in {"true", "yes", "passed", "pass"}
    return bool(value)


def _validate_acyclic(steps: list[ExecutionStep]) -> None:
    step_by_id = {step.id: step for step in steps}
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(step_id: str) -> None:
        if step_id in visited:
            return
        if step_id in visiting:
            raise ValueError(f"dependency cycle includes {step_id}")
        visiting.add(step_id)
        for dependency in step_by_id[step_id].depends_on:
            visit(dependency)
        visiting.remove(step_id)
        visited.add(step_id)

    for step in steps:
        visit(step.id)


def _compact(text: str, limit: int = 500) -> str:
    compacted = " ".join(text.split())
    if len(compacted) <= limit:
        return compacted
    return compacted[:limit].rstrip() + "..."


class PrefixedConsole:
    def __init__(self, console: Console, prefix: str) -> None:
        self.console = console
        self.prefix = prefix

    def progress(self, message: str) -> None:
        if message == "Thinking....":
            return
        self.console.progress(f"{self.prefix}: {message}")

    def tool_request(self, name: str, arguments: dict[str, Any]) -> None:
        self.console.tool_request(f"{self.prefix}.{name}", arguments)

    def tool_result(self, name: str, ok: bool, content: str, diff: str = "") -> None:
        self.console.tool_result(f"{self.prefix}.{name}", ok, content, diff)

    def answer(self, content: str) -> None:
        return

    def token_usage(self, usage: TokenUsage) -> None:
        self.console.token_usage(usage)


class ReadOnlyTools:
    def __init__(self, base_tools: ToolSet) -> None:
        self.base_tools = base_tools

    def schemas(self) -> list[dict[str, Any]]:
        read_only_names = self.base_tools.read_only_tool_names()
        return [
            schema
            for schema in self.base_tools.schemas()
            if schema.get("function", {}).get("name") in read_only_names
        ]

    def invoke(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        if name in self.base_tools.read_only_tool_names():
            return self.base_tools.invoke(name, arguments)
        return ToolResult(
            ok=False,
            content=f"tool is not available in read-only subagent mode: {name}",
        )

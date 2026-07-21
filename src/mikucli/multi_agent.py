from __future__ import annotations

from pathlib import Path
from threading import Lock
from typing import Any

from .llm import BigModelClient
from .logs import RunLog, RunLogWriter, new_session_id
from .memory import LongTermMemory, SessionMemory
from .observability import TraceRecorder, create_trace_recorder
from .orchestration import (
    DEFAULT_SUBAGENTS,
    ExecutionStep,
    PrefixedConsole,
    ReadOnlyTools,
    ReviewDecision,
    SerializedMutationTools,
    StepExecutor,
    StepStatus,
    SubAgentSpec,
    orchestrator_system_prompt,
    parse_execution_plan,
    parse_review_decision,
    subagent_system_prompt,
    summarize_execution,
)
from .react import AgentSession, Console, SessionResult, ToolSet


class OrchestratorSession:
    """Plan a task and coordinate SubAgent execution through focused services."""

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
        self._step_executor = StepExecutor(self)
        self.team_tools = SerializedMutationTools(tools)
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
                tools=self.team_tools if spec.role == "worker" else ReadOnlyTools(self.team_tools),
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
        final_answer = ""
        trace_status = "ok"
        trace_attributes: dict[str, Any] = {}
        try:
            run_log.add_event("agent_started", agent="orchestrator")
            self.memory.add_conversation({"role": "user", "content": task_prompt}, content=task_prompt)
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
                        span_attributes={
                            "subagent.id": self._subagent_id_by_role("planner"),
                            "subagent.role": "planner",
                        },
                        session_mode="multi_agent",
                    )
                    self.trace_recorder.end_span(
                        plan_span_id,
                        attributes={"planner.answer.length": len(planner_result.final_answer)},
                    )
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
                            steps=[
                                {"id": step.id, "title": step.title, "depends_on": step.depends_on}
                                for step in steps
                            ],
                        )
                        self._show_plan(steps)
                        self.console.progress("phase 2: executing")
                        self._execute_steps(
                            task_prompt,
                            steps,
                            run_log,
                            trace_id=trace_id,
                            workflow_span_id=workflow_span_id,
                        )
                        final_answer = summarize_execution(steps)
                except BaseException as exc:
                    self.trace_recorder.end_span(
                        plan_span_id,
                        status="error",
                        attributes={"error.type": type(exc).__name__, "error.message": str(exc)},
                    )
                    raise
            except ValueError as exc:
                trace_status = "error"
                trace_attributes.update({"error.type": type(exc).__name__, "error.message": str(exc)})
                final_answer = f"Could not execute the orchestrator workflow: {exc}"
                run_log.add_event("workflow_failed", error=str(exc))

            self.memory.add_fact(final_answer)
            self.memory.add_conversation({"role": "assistant", "content": final_answer}, content=final_answer)
            self.console.answer(final_answer)
            run_log.final_answer = final_answer
            log_path = self.log_writer.write(run_log)
            return SessionResult(final_answer=final_answer, log_path=log_path)
        except BaseException as exc:
            trace_status = "error"
            trace_attributes.update({"error.type": type(exc).__name__, "error.message": str(exc)})
            raise
        finally:
            trace_attributes["final_answer.length"] = len(final_answer)
            self.trace_recorder.end_span(workflow_span_id, status=trace_status, attributes=trace_attributes)
            self.trace_recorder.end_span(agent_span_id, status=trace_status, attributes=trace_attributes)
            self.trace_recorder.end_trace(trace_id, status=trace_status, attributes=trace_attributes)

    def _execute_steps(
        self,
        task_prompt: str,
        steps: list[ExecutionStep],
        run_log: RunLog,
        *,
        trace_id: str = "",
        workflow_span_id: str = "",
    ) -> None:
        self._step_executor.execute_steps(
            task_prompt,
            steps,
            run_log,
            trace_id=trace_id,
            workflow_span_id=workflow_span_id,
        )

    def _show_plan(self, steps: list[ExecutionStep]) -> None:
        self.console.progress("plan:")
        for step in steps:
            self.console.progress(f"{step.id}: {step.title or step.task}")

    def _planner(self) -> AgentSession:
        return self._subagent_by_role("planner")

    def _reviewer(self) -> AgentSession:
        return self._subagent_by_role("reviewer")

    def _workers(self) -> list[tuple[str, AgentSession]]:
        workers = [
            (spec.id, self.subagents[spec.id])
            for spec in self.subagent_specs
            if spec.role == "worker"
        ]
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

    @staticmethod
    def _planner_task(task_prompt: str) -> str:
        return "Create the execution plan for this task. Return only the required JSON object.\n\n" f"Task:\n{task_prompt}"

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
        return self._step_executor.dependency_context(step)

    @staticmethod
    def _review_task(task_prompt: str, step: ExecutionStep, worker_result: str) -> str:
        return (
            "Review this completed execution step. Return only the required JSON review object.\n\n"
            f"Original task:\n{task_prompt}\n\n"
            f"ExecutionStep {step.id}: {step.title or step.id}\n"
            f"{step.task}\n\n"
            f"Worker result:\n{worker_result}"
        )


__all__ = [
    "DEFAULT_SUBAGENTS",
    "ExecutionStep",
    "OrchestratorSession",
    "PrefixedConsole",
    "ReadOnlyTools",
    "ReviewDecision",
    "SerializedMutationTools",
    "StepStatus",
    "SubAgentSpec",
    "orchestrator_system_prompt",
    "parse_execution_plan",
    "parse_review_decision",
    "subagent_system_prompt",
    "summarize_execution",
]

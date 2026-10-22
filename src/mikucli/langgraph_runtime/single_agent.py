from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime

from mikucli.agent_runtime.cancellation import StopRequested, binds_stop_context, raise_if_stop_requested
from mikucli.agent_runtime.messages import assistant_message as _assistant_message
from mikucli.agent_runtime.tool_execution import handle_tool_calls
from mikucli.agent_runtime.outcomes import finish_session, trace_outcome
from mikucli.run_control import call_chat, managed_turn
from mikucli.json_actions import parse_json_action
from mikucli.llm import ToolCall
from mikucli.logs import RunLog, new_session_id
from mikucli.react import AgentSession, SessionResult
from mikucli.skills import Skill


class GraphAgentState(TypedDict):
    """Serializable control state for one LangGraph-backed agent turn."""

    task_prompt: str
    step_count: int
    tool_calls: list[dict[str, Any]]
    native_tool_calls: bool
    final_answer: str
    trace_status: str


@dataclass(frozen=True)
class TurnContext:
    """Non-checkpointed services and metadata used while a turn executes."""

    run_log: RunLog
    active_skill: Skill | None
    trace_id: str
    agent_span_id: str
    stop_requested: Callable[[], bool] | None


class GraphAgentSession(AgentSession):
    """Run MikuCLI's single-agent tool loop through an explicit LangGraph."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.graph = self._build_graph()

    def _build_graph(self):
        builder = StateGraph(GraphAgentState, context_schema=TurnContext)
        builder.add_node("call_model", self._call_model)
        builder.add_node("execute_tools", self._execute_tools)
        builder.add_node("finalize", self._finalize)
        builder.add_node("max_steps", self._max_steps)
        builder.add_conditional_edges(
            START,
            self._route_from_start,
            {"call_model": "call_model", "max_steps": "max_steps"},
        )
        builder.add_conditional_edges(
            "call_model",
            self._route_after_model,
            {"execute_tools": "execute_tools", "finalize": "finalize"},
        )
        builder.add_conditional_edges(
            "execute_tools",
            self._route_after_tools,
            {"call_model": "call_model", "max_steps": "max_steps"},
        )
        builder.add_edge("finalize", END)
        builder.add_edge("max_steps", END)
        return builder.compile()

    @managed_turn
    @binds_stop_context
    def run_turn(
        self,
        task_prompt: str,
        *,
        active_skill: Skill | None = None,
        trace_id: str = "",
        parent_span_id: str | None = None,
        span_name: str = "agent.session",
        span_kind: str = "agent",
        span_attributes: dict[str, Any] | None = None,
        session_mode: str = "single_agent",
        stop_requested: Callable[[], bool] | None = None,
    ) -> SessionResult:
        run_log = RunLog(
            session_id=new_session_id(),
            task_prompt=task_prompt,
            model=self.model,
            workspace=str(self.workspace),
        )
        run_log.metadata["runtime"] = "langgraph"
        skill_attributes = active_skill.telemetry_attributes() if active_skill is not None else {}
        if active_skill is not None:
            run_log.metadata["skill"] = active_skill.log_metadata()
        owns_trace = not trace_id
        if owns_trace:
            trace_id = self.trace_recorder.start_trace(
                run_id=run_log.session_id,
                task_prompt=task_prompt,
                workspace=str(self.workspace),
                model=self.model,
                session_mode=session_mode,
                attributes={
                    "agent.name": self.agent_name,
                    "runtime.engine": "langgraph",
                    **skill_attributes,
                },
            )
        if trace_id:
            run_log.metadata["trace_id"] = trace_id
        agent_span_id = self.trace_recorder.start_span(
            trace_id=trace_id,
            name=span_name,
            kind=span_kind,
            parent_span_id=parent_span_id,
            attributes={
                "agent.name": self.agent_name,
                "model": self.model,
                "workspace": str(self.workspace),
                "runtime.engine": "langgraph",
                **skill_attributes,
                **(span_attributes or {}),
            },
        )
        final_answer = ""
        trace_status = "ok"
        trace_attributes: dict[str, Any] = {}
        try:
            run_log.add_event("agent_started", agent=self.agent_name)
            self.memory.add_conversation({"role": "user", "content": task_prompt}, content=task_prompt)
            run_log.add_event("user_message", content=task_prompt)
            result = self.graph.invoke(
                {
                    "task_prompt": task_prompt,
                    "step_count": 0,
                    "tool_calls": [],
                    "native_tool_calls": True,
                    "final_answer": "",
                    "trace_status": "ok",
                },
                config={"recursion_limit": self.max_steps * 2 + 3},
                context=TurnContext(
                    run_log=run_log,
                    active_skill=active_skill,
                    trace_id=trace_id,
                    agent_span_id=agent_span_id,
                    stop_requested=stop_requested,
                ),
            )
            final_answer = result["final_answer"]
            trace_status = result["trace_status"]
            return finish_session(self, run_log, final_answer, trace_status)
        except (StopRequested, KeyboardInterrupt):
            trace_status = "stopped"
            final_answer = "Stopped by user."
            run_log.add_event("agent_stopped", reason="user_request")
            self.memory.add_conversation({"role": "assistant", "content": final_answer}, content=final_answer)
            self.console.answer(final_answer)
            return finish_session(self, run_log, final_answer, trace_status)
        except Exception as exc:
            trace_status = "error"
            trace_attributes.update({"error.type": type(exc).__name__, "error.message": str(exc)})
            final_answer = f"Task failed: {exc}"
            self.console.answer(final_answer)
            return finish_session(self, run_log, final_answer, trace_status, exc)
        except BaseException as exc:
            trace_status = "error"
            trace_attributes.update({"error.type": type(exc).__name__, "error.message": str(exc)})
            raise
        finally:
            run_log.final_answer = final_answer
            trace_status = trace_outcome(run_log, trace_status, trace_attributes)
            attributes = {
                "final_answer.length": len(final_answer),
                "changed_paths": run_log.changed_paths,
                **trace_attributes,
            }
            self.trace_recorder.end_span(agent_span_id, status=trace_status, attributes=attributes)
            if owns_trace:
                self.trace_recorder.end_trace(trace_id, status=trace_status, attributes=attributes)

    def _call_model(self, state: GraphAgentState, runtime: Runtime[TurnContext]) -> dict[str, Any]:
        context = runtime.context
        raise_if_stop_requested(context.stop_requested)
        self.console.progress("Thinking....")
        messages = self.memory.messages(
            query=state["task_prompt"],
            system_overlay=(
                context.active_skill.system_overlay() if context.active_skill is not None else ""
            ),
        )
        tool_schemas = self.tools.schemas()
        turn_index = state["step_count"]
        llm_span_id = self.trace_recorder.start_span(
            trace_id=context.trace_id,
            name="llm.chat",
            kind="llm",
            parent_span_id=context.agent_span_id,
            attributes={
                "agent.name": self.agent_name,
                "model": self.model,
                "turn.index": turn_index,
                "llm.message_count": len(messages),
                "llm.tool_schema_count": len(tool_schemas),
                "runtime.engine": "langgraph",
            },
        )
        try:
            assistant = call_chat(self.client, model=self.model, messages=messages, tools=tool_schemas)
        except BaseException as exc:
            self.trace_recorder.end_span(
                llm_span_id,
                status="error",
                attributes={"error.type": type(exc).__name__, "error.message": str(exc)},
            )
            raise
        self.trace_recorder.end_span(
            llm_span_id,
            attributes={
                "llm.prompt_tokens": assistant.token_usage.prompt_tokens,
                "llm.completion_tokens": assistant.token_usage.completion_tokens,
                "llm.total_tokens": assistant.token_usage.total_tokens,
                "llm.tool_call_count": len(assistant.tool_calls),
                "llm.content.length": len(assistant.content),
            },
        )
        raise_if_stop_requested(context.stop_requested)
        self.console.token_usage(assistant.token_usage)
        self._maybe_compress_context(
            assistant.token_usage,
            context.run_log,
            trace_id=context.trace_id,
            parent_span_id=context.agent_span_id,
        )
        context.run_log.add_event(
            "assistant_message",
            content=assistant.content,
            tool_calls=[call.__dict__ for call in assistant.tool_calls],
            token_usage=assistant.token_usage.__dict__,
        )

        if assistant.tool_calls:
            self.memory.add_conversation(
                _assistant_message(assistant.content, assistant.tool_calls),
                content=assistant.content,
            )
            return {
                "step_count": turn_index + 1,
                "tool_calls": [call.__dict__ for call in assistant.tool_calls],
                "native_tool_calls": True,
            }

        fallback = parse_json_action(assistant.content)
        if fallback and fallback.kind == "tool":
            call = ToolCall(id="json_fallback", name=fallback.name, arguments=fallback.arguments)
            self.memory.add_conversation(
                {"role": "assistant", "content": assistant.content},
                content=assistant.content,
            )
            return {
                "step_count": turn_index + 1,
                "tool_calls": [call.__dict__],
                "native_tool_calls": False,
            }

        final_answer = (
            fallback.arguments["content"] if fallback and fallback.kind == "final" else assistant.content
        )
        return {
            "step_count": turn_index + 1,
            "tool_calls": [],
            "final_answer": final_answer,
        }

    def _execute_tools(self, state: GraphAgentState, runtime: Runtime[TurnContext]) -> dict[str, Any]:
        context = runtime.context
        calls = [ToolCall(**call) for call in state["tool_calls"]]
        handle_tool_calls(
            self,
            calls,
            context.run_log,
            native=state["native_tool_calls"],
            trace_id=context.trace_id,
            parent_span_id=context.agent_span_id,
            stop_requested=context.stop_requested,
        )
        return {"tool_calls": []}

    def _finalize(self, state: GraphAgentState) -> dict[str, Any]:
        final_answer = state["final_answer"]
        self.memory.add_conversation(
            {"role": "assistant", "content": final_answer},
            content=final_answer,
        )
        self.console.answer(final_answer)
        return {"trace_status": "ok"}

    def _max_steps(self, state: GraphAgentState) -> dict[str, Any]:
        del state
        final_answer = "Stopped because the session reached the maximum tool loop depth."
        self.memory.add_conversation(
            {"role": "assistant", "content": final_answer},
            content=final_answer,
        )
        self.console.answer(final_answer)
        return {"final_answer": final_answer, "trace_status": "max_steps"}

    def _route_from_start(self, state: GraphAgentState) -> Literal["call_model", "max_steps"]:
        del state
        return "call_model" if self.max_steps > 0 else "max_steps"

    @staticmethod
    def _route_after_model(state: GraphAgentState) -> Literal["execute_tools", "finalize"]:
        return "execute_tools" if state["tool_calls"] else "finalize"

    def _route_after_tools(self, state: GraphAgentState) -> Literal["call_model", "max_steps"]:
        return "max_steps" if state["step_count"] >= self.max_steps else "call_model"


__all__ = ["GraphAgentSession", "GraphAgentState", "TurnContext"]

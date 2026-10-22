from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from .agent_runtime.cancellation import (
    StopRequested,
    binds_stop_context,
    raise_if_stop_requested,
)
from .agent_runtime.contracts import (
    BASE_AGENT_INSTRUCTIONS,
    SYSTEM_PROMPT,
    Console,
    SessionResult,
    ToolSet,
)
from .agent_runtime.messages import assistant_message as _assistant_message
from .agent_runtime.outcomes import finish_session, trace_outcome
from .agent_runtime.tool_execution import handle_tool_calls
from .json_actions import parse_json_action
from .llm import ModelClient, TokenUsage, ToolCall
from .logs import RunLog, RunLogWriter, new_session_id
from .memory import (
    LongTermMemory,
    MapReduceContextCompressor,
    MemoryRetriever,
    SessionMemory,
    token_usage_ratio,
)
from .memory.archive import SessionArchive
from .memory.prompt_budget import check_prompt, estimate_prompt_tokens, output_reserve
from .memory.tools import SessionMemoryTools
from .observability import TraceRecorder, create_trace_recorder
from .run_control import RunLimitExceeded, RunLimits, call_chat, managed_turn
from .skills import Skill


class AgentSession:
    def __init__(
        self,
        *,
        client: ModelClient,
        model: str,
        workspace: Path,
        tools: ToolSet,
        console: Console,
        max_steps: int = 30,
        context_window_tokens: int = 128000,
        memory_window_entries: int = 40,
        compression_threshold: float = 0.8,
        long_term_memory: LongTermMemory | None = None,
        memory_retriever: MemoryRetriever | None = None,
        retain_recent_rounds: int = 3,
        system_prompt: str = SYSTEM_PROMPT,
        agent_name: str = "agent",
        trace_recorder: TraceRecorder | None = None,
        run_limits: RunLimits | None = None,
    ) -> None:
        self.client = client
        self.model = model
        self.workspace = workspace
        self.tools = tools
        self.console = console
        self.max_steps = max_steps
        if max_steps <= 0:
            raise ValueError("max_steps must be positive")
        self.run_limits = run_limits or RunLimits()
        if context_window_tokens <= 0:
            raise ValueError("context_window_tokens must be positive")
        if compression_threshold <= 0:
            raise ValueError("compression_threshold must be positive")
        self.context_window_tokens = context_window_tokens
        self.compression_threshold = compression_threshold
        self.retain_recent_rounds = retain_recent_rounds
        self.long_term_memory = long_term_memory
        self.agent_name = agent_name
        self.trace_recorder = trace_recorder or create_trace_recorder(workspace)
        self.memory = SessionMemory(
            system_message={"role": "system", "content": system_prompt},
            max_active_entries=memory_window_entries,
            long_term_memory=long_term_memory,
            retriever=memory_retriever,
            archive=SessionArchive(workspace / ".mikucli" / "session_memory.sqlite3"),
        )
        self.context_compressor = MapReduceContextCompressor(
            client=client,
            model=model,
            long_term_memory=long_term_memory,
            chunk_chars=min(6000, max(256, (context_window_tokens - output_reserve(context_window_tokens) - 2500) // 2)),
        )
        self.model_tools = SessionMemoryTools(tools, self.memory)
        self.log_writer = RunLogWriter(workspace)

    def clear_chat_history(self) -> None:
        self.memory.clear()

    def request_stop(self) -> None:
        stop = getattr(self.tools, "stop_current_process", None)
        if callable(stop):
            stop()

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
        run_log.metadata["memory_scope_id"] = self.memory.scope_id
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
                attributes={"agent.name": self.agent_name, **skill_attributes},
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
            for turn_index in range(self.max_steps):
                raise_if_stop_requested(stop_requested)
                self.console.progress("Thinking....")
                tool_schemas = self.model_tools.schemas()
                messages = self._prepare_model_messages(task_prompt, active_skill, tool_schemas,
                                                        run_log, trace_id, agent_span_id)
                llm_span_id = self.trace_recorder.start_span(
                    trace_id=trace_id,
                    name="llm.chat",
                    kind="llm",
                    parent_span_id=agent_span_id,
                    attributes={
                        "agent.name": self.agent_name,
                        "model": self.model,
                        "turn.index": turn_index,
                        "llm.message_count": len(messages),
                        "llm.tool_schema_count": len(tool_schemas),
                    },
                )
                try:
                    assistant = call_chat(self.client,
                        model=self.model,
                        messages=messages,
                        tools=tool_schemas,
                    )
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
                raise_if_stop_requested(stop_requested)
                self.console.token_usage(assistant.token_usage)
                self._maybe_compress_context(assistant.token_usage, run_log, trace_id=trace_id, parent_span_id=agent_span_id)
                run_log.add_event(
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
                    handle_tool_calls(
                        self,
                        assistant.tool_calls,
                        run_log,
                        native=True,
                        trace_id=trace_id,
                        parent_span_id=agent_span_id,
                        stop_requested=stop_requested,
                    )
                    continue

                fallback = parse_json_action(assistant.content)
                if fallback and fallback.kind == "tool":
                    call = ToolCall(id="json_fallback", name=fallback.name, arguments=fallback.arguments)
                    self.memory.add_conversation({"role": "assistant", "content": assistant.content}, content=assistant.content)
                    handle_tool_calls(
                        self,
                        [call],
                        run_log,
                        native=False,
                        trace_id=trace_id,
                        parent_span_id=agent_span_id,
                        stop_requested=stop_requested,
                    )
                    continue

                final_answer = fallback.arguments["content"] if fallback and fallback.kind == "final" else assistant.content
                self.memory.add_conversation({"role": "assistant", "content": final_answer}, content=final_answer)
                self.console.answer(final_answer)
                break
            else:
                final_answer = "Stopped because the session reached the maximum tool loop depth."
                self.memory.add_conversation({"role": "assistant", "content": final_answer}, content=final_answer)
                self.console.answer(final_answer)
                trace_status = "max_steps"
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
            self.trace_recorder.end_span(
                agent_span_id,
                status=trace_status,
                attributes={
                    "final_answer.length": len(final_answer),
                    "changed_paths": run_log.changed_paths,
                    **trace_attributes,
                },
            )
            if owns_trace:
                self.trace_recorder.end_trace(
                    trace_id,
                    status=trace_status,
                    attributes={
                        "final_answer.length": len(final_answer),
                        "changed_paths": run_log.changed_paths,
                        **trace_attributes,
                    },
                )

    def _finish_run_log(self, run_log: RunLog, final_answer: str) -> Path:
        run_log.final_answer = final_answer
        return self.log_writer.write(run_log)

    def _prepare_model_messages(self, task_prompt: str, active_skill: Skill | None,
                                tool_schemas: list[dict], run_log: RunLog,
                                trace_id: str = "", parent_span_id: str = "") -> list[dict]:
        overlay = active_skill.system_overlay() if active_skill is not None else ""
        required = self.memory.messages(system_overlay=overlay, include_retrieved=False)
        ledger = self.memory.ledger.message()
        check_prompt([required[0], *([ledger] if ledger else [])], tool_schemas, self.context_window_tokens)
        try:
            check_prompt(required, tool_schemas, self.context_window_tokens)
        except RunLimitExceeded:
            # Preflight compression uses the same checked commit as usage-triggered compression.
            usage = TokenUsage(total_tokens=estimate_prompt_tokens(required, tool_schemas))
            self._maybe_compress_context(usage, run_log, trace_id=trace_id, parent_span_id=parent_span_id,
                                         force=True)
        return self.memory.messages(query=task_prompt, system_overlay=overlay,
                                    context_window_tokens=self.context_window_tokens, tools=tool_schemas)

    def _maybe_compress_context(
        self,
        usage: TokenUsage,
        run_log: RunLog,
        *,
        trace_id: str = "",
        parent_span_id: str = "",
        force: bool = False,
    ) -> None:
        ratio = token_usage_ratio(usage.total_tokens, self.context_window_tokens)
        if ratio is None or (not force and ratio <= self.compression_threshold):
            return
        span_id = self.trace_recorder.start_span(
            trace_id=trace_id,
            name="memory.compress",
            kind="memory",
            parent_span_id=parent_span_id,
            attributes={
                "memory.token_usage_ratio": ratio,
                "llm.total_tokens": usage.total_tokens,
                "context_window_tokens": self.context_window_tokens,
            },
        )
        status = "ok"
        span_attributes: dict[str, Any] = {}
        try:
            self._compress_context(usage, run_log, ratio, span_attributes)
        except BaseException as exc:
            status = "error"
            span_attributes.update({"error.type": type(exc).__name__, "error.message": str(exc)})
            raise
        finally:
            self.trace_recorder.end_span(span_id, status=status, attributes=span_attributes)

    def _compress_context(
        self,
        usage: TokenUsage,
        run_log: RunLog,
        ratio: float,
        span_attributes: dict[str, Any],
    ) -> None:
        compressible_entry_count = self.memory.prepare_entries_for_compression(self.retain_recent_rounds)
        summary = self.memory.compress_old_entries(self.context_compressor)
        if summary is None:
            span_attributes["memory.compressed"] = False
            span_attributes["memory.compressible_entry_count"] = compressible_entry_count
            return
        source_entry_count = int(summary.metadata.get("source_entry_count", 0))
        map_chunk_count = int(summary.metadata.get("map_chunk_count", 0))
        saved_fact_count = int(summary.metadata.get("saved_fact_count", 0))
        span_attributes.update(
            {
                "memory.compressed": True,
                "memory.compressible_entry_count": compressible_entry_count,
                "memory.source_entry_count": source_entry_count,
                "memory.map_chunk_count": map_chunk_count,
                "memory.saved_fact_count": saved_fact_count,
            }
        )
        run_log.add_event(
            "memory_compressed",
            token_usage_ratio=ratio,
            compressible_entry_count=compressible_entry_count,
            source_entry_count=source_entry_count,
            map_chunk_count=map_chunk_count,
            saved_fact_count=saved_fact_count,
            archive_batch_id=summary.metadata.get("archive_batch_id"),
            protected_record_count=len(self.memory.ledger.records),
        )
        self.console.progress(
            f"Compressed {source_entry_count} old session memory entries "
            f"across {map_chunk_count} chunk(s); saved {saved_fact_count} long-term fact(s)."
        )

__all__ = [
    "AgentSession",
    "BASE_AGENT_INSTRUCTIONS",
    "Console",
    "SYSTEM_PROMPT",
    "SessionResult",
    "ToolSet",
]

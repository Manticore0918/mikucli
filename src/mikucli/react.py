from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from .json_actions import parse_json_action
from .llm import BigModelClient, TokenUsage, ToolCall
from .logs import RunLog, RunLogWriter, new_session_id
from .memory import LongTermMemory, MapReduceContextCompressor, SessionMemory, token_usage_ratio
from .observability import TraceRecorder, create_trace_recorder


BASE_AGENT_INSTRUCTIONS = """Use tools only when needed. Do not reveal raw internal reasoning or chain-of-thought.
When you need a tool, use native tool calling if available. If native tool calling is not available, emit only strict JSON:
{"tool": "read_file", "arguments": {"path": "README.md"}}
When the user shares a durable preference or fact that should help future sessions, use save_long_term_memory.
Use search_codebase when you need to discover project structure, symbol behavior, implementation locations, or cross-file relationships.
Use read_file after retrieval when you need exact line-level source inspection.
When read_file reports that a file is too large, use search_codebase to locate relevant passages when available, then choose start_line and end_line for bounded exact reads. Do not retry the same unbounded read.

When you can answer the user, respond normally and concisely.
"""


SYSTEM_PROMPT = f"""You are mikucli, a local command-line agent runner.

{BASE_AGENT_INSTRUCTIONS}
"""


class ToolSet(Protocol):
    def schemas(self) -> list[dict[str, Any]]: ...
    def invoke(self, name: str, arguments: dict[str, Any]) -> Any: ...
    def read_only_tool_names(self) -> set[str]: ...
    def requires_approval(self, name: str) -> bool: ...


class Console(Protocol):
    def progress(self, message: str) -> None: ...
    def tool_request(self, name: str, arguments: dict[str, Any]) -> None: ...
    def tool_result(self, name: str, ok: bool, content: str, diff: str = "") -> None: ...
    def answer(self, content: str) -> None: ...
    def token_usage(self, usage: TokenUsage) -> None: ...


@dataclass
class SessionResult:
    final_answer: str
    log_path: Path


class AgentSession:
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
        system_prompt: str = SYSTEM_PROMPT,
        agent_name: str = "agent",
        trace_recorder: TraceRecorder | None = None,
    ) -> None:
        self.client = client
        self.model = model
        self.workspace = workspace
        self.tools = tools
        self.console = console
        self.max_steps = max_steps
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
        )
        self.context_compressor = MapReduceContextCompressor(
            client=client,
            model=model,
            long_term_memory=long_term_memory,
        )
        self.log_writer = RunLogWriter(workspace)

    def clear_chat_history(self) -> None:
        self.memory.active_entries.clear()
        self.memory.old_entries.clear()
        self.memory.summary_entries.clear()

    def run_turn(
        self,
        task_prompt: str,
        *,
        trace_id: str = "",
        parent_span_id: str | None = None,
        span_name: str = "agent.session",
        span_kind: str = "agent",
        span_attributes: dict[str, Any] | None = None,
        session_mode: str = "single_agent",
    ) -> SessionResult:
        run_log = RunLog(
            session_id=new_session_id(),
            task_prompt=task_prompt,
            model=self.model,
            workspace=str(self.workspace),
        )
        owns_trace = not trace_id
        if owns_trace:
            trace_id = self.trace_recorder.start_trace(
                run_id=run_log.session_id,
                task_prompt=task_prompt,
                workspace=str(self.workspace),
                model=self.model,
                session_mode=session_mode,
                attributes={"agent.name": self.agent_name},
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
                **(span_attributes or {}),
            },
        )
        run_log.add_event("agent_started", agent=self.agent_name)
        self.memory.add_conversation({"role": "user", "content": task_prompt}, content=task_prompt)
        run_log.add_event("user_message", content=task_prompt)

        final_answer = ""
        trace_status = "ok"
        try:
            for turn_index in range(self.max_steps):
                self.console.progress("Thinking....")
                messages = self.memory.messages(query=task_prompt)
                tool_schemas = self.tools.schemas()
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
                    assistant = self.client.chat(
                        model=self.model,
                        messages=messages,
                        tools=tool_schemas,
                    )
                except Exception as exc:
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
                    self._handle_tool_calls(assistant.tool_calls, run_log, native=True, trace_id=trace_id, parent_span_id=agent_span_id)
                    continue

                fallback = parse_json_action(assistant.content)
                if fallback and fallback.kind == "tool":
                    call = ToolCall(id="json_fallback", name=fallback.name, arguments=fallback.arguments)
                    self.memory.add_conversation({"role": "assistant", "content": assistant.content}, content=assistant.content)
                    self._handle_tool_calls([call], run_log, native=False, trace_id=trace_id, parent_span_id=agent_span_id)
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
            return SessionResult(final_answer=final_answer, log_path=self._finish_run_log(run_log, final_answer))
        except Exception:
            trace_status = "error"
            raise
        finally:
            run_log.final_answer = final_answer
            self.trace_recorder.end_span(
                agent_span_id,
                status=trace_status,
                attributes={
                    "final_answer.length": len(final_answer),
                    "changed_paths": run_log.changed_paths,
                },
            )
            if owns_trace:
                self.trace_recorder.end_trace(
                    trace_id,
                    status=trace_status,
                    attributes={
                        "final_answer.length": len(final_answer),
                        "changed_paths": run_log.changed_paths,
                    },
                )

    def _finish_run_log(self, run_log: RunLog, final_answer: str) -> Path:
        run_log.final_answer = final_answer
        return self.log_writer.write(run_log)

    def _maybe_compress_context(
        self,
        usage: TokenUsage,
        run_log: RunLog,
        *,
        trace_id: str = "",
        parent_span_id: str = "",
    ) -> None:
        ratio = token_usage_ratio(usage.total_tokens, self.context_window_tokens)
        if ratio is None or ratio <= self.compression_threshold:
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
        except Exception as exc:
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
        )
        self.console.progress(
            f"Compressed {source_entry_count} old session memory entries "
            f"across {map_chunk_count} chunk(s); saved {saved_fact_count} long-term fact(s)."
        )

    def _handle_tool_calls(
        self,
        calls: list[ToolCall],
        run_log: RunLog,
        native: bool,
        *,
        trace_id: str = "",
        parent_span_id: str = "",
    ) -> None:
        for call in calls:
            self.console.tool_request(call.name, call.arguments)
            span_id = self.trace_recorder.start_span(
                trace_id=trace_id,
                name="tool.invoke",
                kind="tool",
                parent_span_id=parent_span_id,
                attributes={
                    "tool.name": call.name,
                    "tool.call_id": call.id,
                    "tool.native": native,
                    "tool.arguments": _summarize_value(call.arguments),
                },
            )
            try:
                result = self.tools.invoke(call.name, call.arguments)
            except Exception as exc:
                self.trace_recorder.end_span(
                    span_id,
                    status="error",
                    attributes={"tool.name": call.name, "error.type": type(exc).__name__, "error.message": str(exc)},
                )
                raise
            self.trace_recorder.end_span(
                span_id,
                status="ok" if result.ok else "error",
                attributes={
                    "tool.name": call.name,
                    "tool.ok": result.ok,
                    "tool.changed_paths": result.changed_paths,
                    "tool.output": _capture_tool_output(result.content),
                },
            )
            run_log.add_event(
                "tool_result",
                tool=call.name,
                ok=result.ok,
                content=result.content,
                changed_paths=result.changed_paths,
            )
            run_log.add_changed_paths(result.changed_paths)
            self.console.tool_result(call.name, result.ok, result.content, result.diff)
            if native:
                self.memory.add_tool_result(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "content": result.content,
                    },
                    tool_name=call.name,
                    ok=result.ok,
                    content=result.content,
                )
            else:
                content = f"Tool result for {call.name}:\n{result.content}"
                self.memory.add_conversation(
                    {
                        "role": "user",
                        "content": content,
                    },
                    content=content,
                )


def _assistant_message(content: str, tool_calls: list[ToolCall]) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": content,
        "tool_calls": [
            {
                "id": call.id,
                "type": "function",
                "function": {"name": call.name, "arguments": json.dumps(call.arguments)},
            }
            for call in tool_calls
        ],
    }


def _capture_tool_output(content: str, *, limit: int = 500) -> str:
    mode = os.environ.get("MIKUCLI_OBS_CAPTURE_TOOL_OUTPUT", "summary").strip().casefold()
    if mode == "off":
        return ""
    if mode == "full" or len(content) <= limit:
        return content
    return content[:limit].rstrip() + "\n... truncated ..."


def _summarize_value(value: Any, *, limit: int = 500) -> Any:
    if isinstance(value, dict):
        return {str(key): _summarize_value(item, limit=limit) for key, item in value.items()}
    if isinstance(value, list):
        return [_summarize_value(item, limit=limit) for item in value]
    if isinstance(value, tuple):
        return [_summarize_value(item, limit=limit) for item in value]
    if isinstance(value, str):
        if len(value) <= limit:
            return value
        return value[:limit].rstrip() + "\n... truncated ..."
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    return str(value)

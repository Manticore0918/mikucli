from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from .json_actions import parse_json_action
from .llm import BigModelClient, TokenUsage, ToolCall
from .logs import RunLog, RunLogWriter, new_session_id
from .memory import LongTermMemory, MapReduceContextCompressor, SessionMemory, token_usage_ratio


BASE_AGENT_INSTRUCTIONS = """Use tools only when needed. Do not reveal raw internal reasoning or chain-of-thought.
When you need a tool, use native tool calling if available. If native tool calling is not available, emit only strict JSON:
{"tool": "read_file", "arguments": {"path": "README.md"}}
When the user shares a durable preference or fact that should help future sessions, use save_long_term_memory.
Use search_codebase when you need to discover project structure, symbol behavior, implementation locations, or cross-file relationships.
Use read_file after retrieval when you need exact line-level source inspection.

When you can answer the user, respond normally and concisely.
"""


SYSTEM_PROMPT = f"""You are mikucli, a local command-line agent runner.

{BASE_AGENT_INSTRUCTIONS}
"""


class ToolSet(Protocol):
    def schemas(self) -> list[dict[str, Any]]: ...
    def invoke(self, name: str, arguments: dict[str, Any]) -> Any: ...
    def read_only_tool_names(self) -> set[str]: ...


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

    def run_turn(self, task_prompt: str) -> SessionResult:
        run_log = RunLog(
            session_id=new_session_id(),
            task_prompt=task_prompt,
            model=self.model,
            workspace=str(self.workspace),
        )
        run_log.add_event("agent_started", agent=self.agent_name)
        self.memory.add_conversation({"role": "user", "content": task_prompt}, content=task_prompt)
        run_log.add_event("user_message", content=task_prompt)

        final_answer = ""
        for _ in range(self.max_steps):
            self.console.progress("Thinking....")
            assistant = self.client.chat(
                model=self.model,
                messages=self.memory.messages(query=task_prompt),
                tools=self.tools.schemas(),
            )
            self.console.token_usage(assistant.token_usage)
            self._maybe_compress_context(assistant.token_usage, run_log)
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
                self._handle_tool_calls(assistant.tool_calls, run_log, native=True)
                continue

            fallback = parse_json_action(assistant.content)
            if fallback and fallback.kind == "tool":
                call = ToolCall(id="json_fallback", name=fallback.name, arguments=fallback.arguments)
                self.memory.add_conversation({"role": "assistant", "content": assistant.content}, content=assistant.content)
                self._handle_tool_calls([call], run_log, native=False)
                continue

            final_answer = fallback.arguments["content"] if fallback and fallback.kind == "final" else assistant.content
            self.memory.add_conversation({"role": "assistant", "content": final_answer}, content=final_answer)
            self.console.answer(final_answer)
            break
        else:
            final_answer = "Stopped because the session reached the maximum tool loop depth."
            self.memory.add_conversation({"role": "assistant", "content": final_answer}, content=final_answer)
            self.console.answer(final_answer)

        run_log.final_answer = final_answer
        log_path = self.log_writer.write(run_log)
        return SessionResult(final_answer=final_answer, log_path=log_path)

    def _maybe_compress_context(self, usage: TokenUsage, run_log: RunLog) -> None:
        ratio = token_usage_ratio(usage.total_tokens, self.context_window_tokens)
        if ratio is None or ratio <= self.compression_threshold:
            return
        compressible_entry_count = self.memory.prepare_entries_for_compression(self.retain_recent_rounds)
        summary = self.memory.compress_old_entries(self.context_compressor)
        if summary is None:
            return
        source_entry_count = int(summary.metadata.get("source_entry_count", 0))
        map_chunk_count = int(summary.metadata.get("map_chunk_count", 0))
        saved_fact_count = int(summary.metadata.get("saved_fact_count", 0))
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

    def _handle_tool_calls(self, calls: list[ToolCall], run_log: RunLog, native: bool) -> None:
        for call in calls:
            self.console.tool_request(call.name, call.arguments)
            result = self.tools.invoke(call.name, call.arguments)
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

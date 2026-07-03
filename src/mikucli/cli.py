from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .codebase.chunking import ChunkingError
from .codebase.embeddings import EmbeddingError
from .codebase.formatting import format_search_results
from .codebase.index import CodebaseIndexError
from .codebase.service import CodebaseService
from .config import Config, load_config
from .console import TerminalConsole
from .llm import BigModelClient
from .mcp_config import McpConfigError, load_mcp_config
from .mcp_tools import McpRuntimeError, McpServerStatus, McpToolSet
from .memory import LongTermMemory, default_long_term_memory_path
from .multi_agent import OrchestratorSession
from .react import AgentSession
from .tools import ToolPolicy, ToolRegistry
from .workspace import Workspace


BANNER_ART = ("mikucli",)


def render_banner() -> str:
    content_width = max(len(line) for line in BANNER_ART)
    inner_width = content_width + 4
    top = "+" + "-" * inner_width + "+"
    bottom = "+" + "-" * inner_width + "+"
    empty = "|" + " " * inner_width + "|"
    art_lines = [f"|  {line.ljust(content_width)}  |" for line in BANNER_ART]
    return "\n".join((top, *art_lines, empty, bottom))


def configure_output_encoding() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mikucli")
    parser.add_argument("task_prompt", nargs="*", help="Initial task prompt for the agent session.")
    parser.add_argument("--workspace", default=".", help="Workspace directory. Defaults to the current directory.")
    parser.add_argument("--model", default=None, help="GLM model name. Defaults to MIKUCLI_MODEL or glm-5.2.")
    parser.add_argument("--env-file", default=None, help="Path to .env file. Defaults to workspace .env.")
    parser.add_argument("--max-steps", type=int, default=30, help="Maximum ReAct tool loop steps per user turn.")
    parser.add_argument(
        "--context-window-tokens",
        type=int,
        default=None,
        help="Token budget used to decide when session memory compression starts.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    configure_output_encoding()
    parser = build_parser()
    args = parser.parse_args(argv)

    print(render_banner())

    workspace = Workspace(Path(args.workspace))
    console = TerminalConsole()

    try:
        env_file = Path(args.env_file) if args.env_file else None
        config = load_config(workspace.root, args.model, env_file, args.context_window_tokens)
    except ValueError as exc:
        print(f"mikucli: {exc}", file=sys.stderr)
        return 2

    client = BigModelClient(api_key=config.api_key, base_url=config.base_url)
    long_term_memory = LongTermMemory(default_long_term_memory_path(config.workspace))
    try:
        codebase_service = CodebaseService(
            workspace=config.workspace,
            embedding_provider=config.embedding_provider,
            embedding_model=config.embedding_model,
            ollama_base_url=config.ollama_base_url,
        )
    except ValueError as exc:
        print(f"mikucli: {exc}", file=sys.stderr)
        return 2

    builtin_tools = ToolRegistry(
        workspace=workspace,
        confirm_tool=console.confirm_tool,
        tool_policy=ToolPolicy(),
        long_term_memory=long_term_memory,
        codebase_service=codebase_service,
    )
    session = _new_single_agent_session(
        client=client,
        config=config,
        tools=builtin_tools,
        console=console,
        max_steps=args.max_steps,
        long_term_memory=long_term_memory,
    )
    team_mode = False
    mcp_tools: McpToolSet | None = None

    initial_prompt = " ".join(args.task_prompt).strip()
    if initial_prompt:
        try:
            if initial_prompt == "/mcp":
                mcp_tools, session = _enable_mcp_mode(
                    client=client,
                    config=config,
                    console=console,
                    max_steps=args.max_steps,
                    long_term_memory=long_term_memory,
                    current_mcp_tools=mcp_tools,
                )
                return 0
            if handle_slash_command(initial_prompt, codebase_service, console):
                return 0
            print(f"You: {initial_prompt}")
            result = session.run_turn(initial_prompt)
            print(f"[log] {result.log_path}")
            return 0
        finally:
            if mcp_tools is not None:
                mcp_tools.close()

    print("mikucli interactive session. Type /team, /mcp, or /exit.")
    try:
        while True:
            try:
                prompt = input("You: ").strip()
            except EOFError:
                print()
                return 0
            if not prompt:
                continue
            if prompt in {"/exit", "/quit"}:
                return 0
            if prompt == "/mcp":
                if mcp_tools is not None:
                    _print_mcp_status(mcp_tools.statuses())
                    continue
                try:
                    mcp_tools, session = _enable_mcp_mode(
                        client=client,
                        config=config,
                        console=console,
                        max_steps=args.max_steps,
                        long_term_memory=long_term_memory,
                        current_mcp_tools=mcp_tools,
                    )
                except (McpConfigError, McpRuntimeError, TimeoutError) as exc:
                    print(f"mikucli: could not enable MCP mode: {exc}", file=sys.stderr)
                    print(f"mikucli: create {config.workspace / '.mikucli' / 'mcp.json'} and try /mcp again.")
                else:
                    team_mode = False
                continue
            if prompt == "/team":
                if team_mode:
                    print("[mode] multi-agent mode is already active.")
                    continue
                if mcp_tools is not None:
                    mcp_tools.close()
                    mcp_tools = None
                session = OrchestratorSession(
                    client=client,
                    model=config.model,
                    workspace=config.workspace,
                    tools=builtin_tools,
                    console=console,
                    max_steps=args.max_steps,
                    context_window_tokens=config.context_window_tokens,
                    long_term_memory=long_term_memory,
                )
                team_mode = True
                print("[mode] multi-agent mode enabled.")
                continue
            if handle_slash_command(prompt, codebase_service, console):
                continue
            result = session.run_turn(prompt)
            print(f"[log] {result.log_path}")
    finally:
        if mcp_tools is not None:
            mcp_tools.close()


def _new_single_agent_session(
    *,
    client: BigModelClient,
    config: Config,
    tools: object,
    console: TerminalConsole,
    max_steps: int,
    long_term_memory: LongTermMemory,
) -> AgentSession:
    return AgentSession(
        client=client,
        model=config.model,
        workspace=config.workspace,
        tools=tools,
        console=console,
        max_steps=max_steps,
        context_window_tokens=config.context_window_tokens,
        long_term_memory=long_term_memory,
    )


def _enable_mcp_mode(
    *,
    client: BigModelClient,
    config: Config,
    console: TerminalConsole,
    max_steps: int,
    long_term_memory: LongTermMemory,
    current_mcp_tools: McpToolSet | None,
) -> tuple[McpToolSet, AgentSession]:
    if current_mcp_tools is not None:
        current_mcp_tools.close()
    mcp_config = load_mcp_config(config.workspace)
    mcp_tools = McpToolSet.connect(
        config=mcp_config,
        workspace=config.workspace,
        confirm_tool=console.confirm_tool,
    )
    _print_mcp_status(mcp_tools.statuses())
    print(f"[mode] MCP mode enabled with {len(mcp_tools.schemas())} tool(s).")
    return mcp_tools, _new_single_agent_session(
        client=client,
        config=config,
        tools=mcp_tools,
        console=console,
        max_steps=max_steps,
        long_term_memory=long_term_memory,
    )


def _print_mcp_status(statuses: list[McpServerStatus]) -> None:
    print("[mcp] server status")
    for status in statuses:
        initialized = "initialized" if status.initialized else "not initialized"
        active = "active" if status.active else "inactive"
        suffix = f" ({status.error})" if status.error else ""
        print(f"[mcp] {status.name}: {initialized}, {active}{suffix}")


def handle_slash_command(prompt: str, codebase_service: CodebaseService, console: TerminalConsole) -> bool:
    if prompt == "/index":
        try:
            codebase_service.rebuild_index(progress=console.progress)
        except (ChunkingError, CodebaseIndexError, EmbeddingError, ValueError) as exc:
            print(f"mikucli: {exc}", file=sys.stderr)
        return True

    if prompt == "/search" or prompt.startswith("/search "):
        query = prompt.removeprefix("/search").strip()
        if not query:
            print("mikucli: usage: /search <natural language query>", file=sys.stderr)
            return True
        try:
            results = codebase_service.search(query, limit=5)
        except (CodebaseIndexError, EmbeddingError) as exc:
            print(f"mikucli: {exc}", file=sys.stderr)
            return True
        print(format_search_results(results, max_content_chars=1000))
        return True

    return False


if __name__ == "__main__":
    raise SystemExit(main())

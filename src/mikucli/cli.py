from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .codebase.chunking import ChunkingError
from .codebase.embeddings import EmbeddingError
from .codebase.formatting import format_search_results
from .codebase.index import CodebaseIndexError
from .codebase.service import CodebaseService
from .config import load_config
from .console import TerminalConsole
from .llm import BigModelClient
from .memory import LongTermMemory, default_long_term_memory_path
from .multi_agent import OrchestratorSession
from .react import AgentSession
from .tools import ToolPolicy, ToolRegistry
from .workspace import Workspace


BANNER_ART = (
    "███╗   ███╗██╗██╗  ██╗██╗   ██╗ ██████╗██╗     ██╗",
    "████╗ ████║██║██║ ██╔╝██║   ██║██╔════╝██║     ██║",
    "██╔████╔██║██║█████╔╝ ██║   ██║██║     ██║     ██║",
    "██║╚██╔╝██║██║██╔═██╗ ██║   ██║██║     ██║     ██║",
    "██║ ╚═╝ ██║██║██║  ██╗╚██████╔╝╚██████╗███████╗██║",
    "╚═╝     ╚═╝╚═╝╚═╝  ╚═╝ ╚═════╝  ╚═════╝╚══════╝╚═╝",
)


def render_banner() -> str:
    inner_width = max(len(line) for line in BANNER_ART) + 6
    top = f"╔{'═' * inner_width}╗"
    bottom = f"╚{'═' * inner_width}╝"
    empty = f"║{' ' * inner_width}║"
    art_lines = [f"║   {line.ljust(inner_width - 6)}   ║" for line in BANNER_ART]
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
    tools = ToolRegistry(
        workspace=workspace,
        confirm_tool=console.confirm_tool,
        tool_policy=ToolPolicy(),
        long_term_memory=long_term_memory,
        codebase_service=codebase_service,
    )
    session = AgentSession(
        client=client,
        model=config.model,
        workspace=config.workspace,
        tools=tools,
        console=console,
        max_steps=args.max_steps,
        context_window_tokens=config.context_window_tokens,
        long_term_memory=long_term_memory,
    )
    team_mode = False

    initial_prompt = " ".join(args.task_prompt).strip()
    if initial_prompt:
        if handle_slash_command(initial_prompt, codebase_service, console):
            return 0
        print(f"👤You: {initial_prompt}")
        result = session.run_turn(initial_prompt)
        print(f"[log] {result.log_path}")
        return 0

    print("mikucli interactive session. Type /team for multi-agent mode or /exit to quit.")
    while True:
        try:
            prompt = input("👤You: ").strip()
        except EOFError:
            print()
            return 0
        if not prompt:
            continue
        if prompt in {"/exit", "/quit"}:
            return 0
        if prompt == "/team":
            if team_mode:
                print("[mode] multi-agent mode is already active.")
                continue
            session = OrchestratorSession(
                client=client,
                model=config.model,
                workspace=config.workspace,
                tools=tools,
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

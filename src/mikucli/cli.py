from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from .codebase.service import CodebaseService
from .cli_support.commands import handle_slash_command
from .cli_support.evaluation import EvalRunController, eval_runner as _eval_runner
from .cli_support.interactive_session import run_interactive_session
from .cli_support.sessions import (
    connect_mcp_tools as _connect_mcp_tools,
    new_session as _new_session,
    new_single_agent_session as _new_single_agent_session,
)
from .config import load_config
from .console import TerminalConsole
from .llm import ModelClient
from .mcp_tools import McpToolSet
from .memory import LongTermMemory, MemoryRetriever, default_long_term_memory_path
from .skills import Skill, SkillError, SkillRegistry
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
    content_width = max(len(line) for line in BANNER_ART)
    inner_width = content_width + 4
    top = "╔" + "═" * inner_width + "╗"
    bottom = "╚" + "═" * inner_width + "╝"
    empty = "║" + " " * inner_width + "║"
    art_lines = [f"║  {line.ljust(content_width)}  ║" for line in BANNER_ART]
    return "\n".join((top, *art_lines, empty, bottom))


def configure_output_encoding() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")


def clear_terminal() -> None:
    """Clear an interactive terminal so the startup banner begins at the top."""
    if not (sys.stdin.isatty() and sys.stdout.isatty()):
        return

    if os.name == "nt":
        os.system("cls")
        return

    print("\033[2J\033[3J\033[H", end="", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mikucli")
    parser.add_argument("task_prompt", nargs="*", help="Initial task prompt for the agent session.")
    parser.add_argument("--workspace", default=".", help="Workspace directory. Defaults to the current directory.")
    parser.add_argument("--model", default=None, help="Model name. Defaults to MIKUCLI_MODEL or glm-5.2.")
    parser.add_argument(
        "--api-key",
        default=None,
        help="Model provider API key. Prefer MIKUCLI_API_KEY to avoid shell history exposure.",
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help="Full OpenAI-compatible Chat Completions URL. Defaults to MIKUCLI_BASE_URL.",
    )
    parser.add_argument("--env-file", default=None, help="Path to an additional .env file with high priority.")
    parser.add_argument("--max-steps", type=int, default=30, help="Maximum ReAct tool loop steps per user turn.")
    parser.add_argument(
        "--runtime",
        choices=("native", "langgraph"),
        default="native",
        help="Single-agent runtime engine. Defaults to native.",
    )
    parser.add_argument(
        "--context-window-tokens",
        type=int,
        default=None,
        help="Token budget used to decide when session memory compression starts.",
    )
    parser.add_argument(
        "--no-tui",
        action="store_true",
        help="Start an interactive session with the inline renderer instead of the full-screen Terminal UI.",
    )
    return parser


def _resolve_task_prompt(prompt: str, skill_registry: SkillRegistry) -> tuple[str, Skill | None]:
    invocation = skill_registry.resolve_prompt(prompt)
    if invocation is None:
        return prompt, None
    return invocation.task_prompt, invocation.skill


def main(argv: list[str] | None = None) -> int:
    configure_output_encoding()
    parser = build_parser()
    args = parser.parse_args(argv)

    clear_terminal()
    print(render_banner())

    workspace = Workspace(Path(args.workspace))
    console = TerminalConsole()

    try:
        env_file = Path(args.env_file) if args.env_file else None
        config = load_config(
            workspace.root,
            args.model,
            env_file,
            args.context_window_tokens,
            api_key=args.api_key,
            base_url=args.base_url,
        )
    except ValueError as exc:
        print(console.error(exc), file=sys.stderr)
        return 2

    client = ModelClient(api_key=config.api_key, base_url=config.base_url)
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
    memory_retriever = MemoryRetriever(
        embedding_client=codebase_service.embedding_client,
        embedding_model=config.embedding_model,
    )

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
        memory_retriever=memory_retriever,
        runtime=args.runtime,
    )
    team_mode = False
    mcp_tools: McpToolSet | None = None
    skill_registry = SkillRegistry(config.workspace)
    eval_controller = EvalRunController(
        _eval_runner(
            client=client,
            config=config,
            max_steps=args.max_steps,
            memory_retriever=memory_retriever,
            runtime=args.runtime,
        )
    )

    initial_prompt = " ".join(args.task_prompt).strip()
    if initial_prompt:
        try:
            if initial_prompt == "/mcp":
                mcp_tools = _connect_mcp_tools(config=config, console=console)
                session = _new_session(
                    client=client,
                    config=config,
                    tools=mcp_tools,
                    console=console,
                    max_steps=args.max_steps,
                    long_term_memory=long_term_memory,
                    memory_retriever=memory_retriever,
                    team_mode=team_mode,
                    runtime=args.runtime,
                )
                console.print_mode(team_mode=team_mode, mcp_enabled=True, tool_count=len(mcp_tools.schemas()))
                return 0
            if initial_prompt == "/team":
                team_mode = True
                session = _new_session(
                    client=client,
                    config=config,
                    tools=builtin_tools,
                    console=console,
                    max_steps=args.max_steps,
                    long_term_memory=long_term_memory,
                    memory_retriever=memory_retriever,
                    team_mode=team_mode,
                    runtime=args.runtime,
                )
                console.print_mode(team_mode=team_mode, mcp_enabled=False, tool_count=len(builtin_tools.schemas()))
                return 0
            if handle_slash_command(
                initial_prompt,
                codebase_service,
                console,
                skill_registry=skill_registry,
                eval_controller=eval_controller,
                eval_background_allowed=False,
                dashboard_workspace=config.workspace,
            ):
                return 0
            print(f"{console.prompt_label()}{initial_prompt}")
            try:
                task_prompt, active_skill = _resolve_task_prompt(initial_prompt, skill_registry)
            except SkillError as exc:
                print(console.error(exc.localized(console.language)), file=sys.stderr)
                return 2
            result = session.run_turn(task_prompt, active_skill=active_skill)
            console.log_path(result.log_path)
            return 0
        finally:
            if mcp_tools is not None:
                mcp_tools.close()

    return run_interactive_session(
        banner=render_banner(),
        client=client,
        config=config,
        console=console,
        codebase_service=codebase_service,
        builtin_tools=builtin_tools,
        session=session,
        long_term_memory=long_term_memory,
        memory_retriever=memory_retriever,
        skill_registry=skill_registry,
        eval_controller=eval_controller,
        max_steps=args.max_steps,
        runtime=args.runtime,
        no_tui=args.no_tui,
        resolve_task_prompt=_resolve_task_prompt,
    )


if __name__ == "__main__":
    raise SystemExit(main())

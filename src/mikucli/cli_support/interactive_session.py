from __future__ import annotations

import sys
from collections.abc import Callable
from contextlib import nullcontext, redirect_stdout
from typing import Any

from prompt_toolkit import PromptSession
from prompt_toolkit.patch_stdout import patch_stdout

from mikucli.cli_support.commands import handle_slash_command
from mikucli.cli_support.evaluation import EvalRunController
from mikucli.cli_support.interactive import ApprovalBroker, InteractiveTurnController
from mikucli.cli_support.sessions import connect_mcp_tools, new_session
from mikucli.codebase.service import CodebaseService
from mikucli.config import Config
from mikucli.console import TerminalConsole
from mikucli.llm import ModelClient
from mikucli.mcp_config import McpConfigError
from mikucli.mcp_tools import McpRuntimeError, McpToolSet
from mikucli.memory import LongTermMemory, MemoryRetriever
from mikucli.skills import Skill, SkillError, SkillRegistry
from mikucli.terminal_ui import FullScreenTerminalUI, TuiAction
from mikucli.tools import ToolRegistry


ResolveTaskPrompt = Callable[[str, SkillRegistry], tuple[str, Skill | None]]


def run_interactive_session(
    *,
    banner: str,
    client: ModelClient,
    config: Config,
    console: TerminalConsole,
    codebase_service: CodebaseService,
    builtin_tools: ToolRegistry,
    session: Any,
    long_term_memory: LongTermMemory,
    memory_retriever: MemoryRetriever,
    skill_registry: SkillRegistry,
    eval_controller: EvalRunController,
    max_steps: int,
    runtime: str,
    no_tui: bool,
    resolve_task_prompt: ResolveTaskPrompt,
) -> int:
    live_prompt = sys.stdin.isatty() and sys.stdout.isatty()
    approval_broker = ApprovalBroker()
    turn_controller = InteractiveTurnController(console)
    console.set_approval_handler(approval_broker.request)
    console.set_status_handler(turn_controller.set_status)
    mode = "inline" if no_tui or not live_prompt else "fullscreen"
    mcp_tools: McpToolSet | None = None
    team_mode = False

    def is_busy() -> bool:
        eval_status = eval_controller.status()
        return turn_controller.is_running() or (eval_status.running and eval_status.exclusive)

    def process_prompt(prompt: str) -> TuiAction:
        nonlocal mcp_tools, mode, session, team_mode

        prompt = prompt.strip()
        if not prompt:
            return "continue"
        if approval_broker.has_pending():
            if prompt == "/stop":
                approval_broker.cancel()
            elif prompt.lower() in {"y", "yes", "n", "no"}:
                approved = prompt.lower() in {"y", "yes"}
                approval_broker.resolve(approved)
                print("mikucli: tool approved." if approved else "mikucli: tool denied.")
                return "continue"
            else:
                print("mikucli: approval required; type y, n, or /stop.")
                return "continue"
        if is_busy() and prompt not in {"/stop", "/eval stop"}:
            print("mikucli: a foreground process is running; type /stop before starting another command.")
            return "continue"
        if prompt in {"/exit", "/quit"}:
            approval_broker.cancel()
            return "exit"
        if prompt == "/tui fullscreen":
            if not live_prompt:
                print("mikucli: full-screen Terminal UI requires interactive stdin and stdout.", file=sys.stderr)
                return "continue"
            if mode == "fullscreen":
                print("mikucli: full-screen Terminal UI is already active.")
                return "continue"
            mode = "fullscreen"
            return "fullscreen"
        if prompt == "/tui inline":
            if mode == "inline":
                print("mikucli: inline Terminal UI mode is already active.")
                return "continue"
            mode = "inline"
            return "inline"
        if prompt == "/tui" or prompt.startswith("/tui "):
            print("mikucli: usage: /tui fullscreen | /tui inline", file=sys.stderr)
            return "continue"
        if prompt == "/mcp":
            if mcp_tools is None:
                try:
                    mcp_tools = connect_mcp_tools(config=config, console=console)
                except (McpConfigError, McpRuntimeError, TimeoutError) as exc:
                    console.print_mcp_enable_error(exc, config.workspace / ".mikucli" / "mcp.json")
                    return "continue"
                active_tools = mcp_tools
                mcp_enabled = True
            else:
                mcp_tools.close()
                mcp_tools = None
                active_tools = builtin_tools
                mcp_enabled = False
            session = new_session(
                client=client,
                config=config,
                tools=active_tools,
                console=console,
                max_steps=max_steps,
                long_term_memory=long_term_memory,
                memory_retriever=memory_retriever,
                team_mode=team_mode,
                runtime=runtime,
            )
            console.print_mode(
                team_mode=team_mode,
                mcp_enabled=mcp_enabled,
                tool_count=len(active_tools.schemas()),
            )
            return "continue"
        if prompt == "/team":
            team_mode = not team_mode
            active_tools = mcp_tools if mcp_tools is not None else builtin_tools
            session = new_session(
                client=client,
                config=config,
                tools=active_tools,
                console=console,
                max_steps=max_steps,
                long_term_memory=long_term_memory,
                memory_retriever=memory_retriever,
                team_mode=team_mode,
                runtime=runtime,
            )
            console.print_mode(
                team_mode=team_mode,
                mcp_enabled=mcp_tools is not None,
                tool_count=len(active_tools.schemas()),
            )
            return "continue"
        if handle_slash_command(
            prompt,
            codebase_service,
            console,
            skill_registry=skill_registry,
            eval_controller=eval_controller,
            eval_background_allowed=True,
            dashboard_workspace=config.workspace,
            stop_handler=turn_controller.request_stop,
        ):
            return "continue"
        try:
            task_prompt, active_skill = resolve_task_prompt(prompt, skill_registry)
        except SkillError as exc:
            print(console.error(exc.localized(console.language)), file=sys.stderr)
            return "continue"
        turn_controller.start(session, task_prompt, active_skill)
        return "continue"

    def status_text() -> str:
        agent_status = turn_controller.status()
        if approval_broker.has_pending():
            agent_status = "approval required"
        parts = [f"Agent: {agent_status}"]
        eval_status = eval_controller.status()
        if eval_status.running:
            eval_state = "stopping" if eval_controller.stop_event.is_set() else "running"
            parts.append(
                f"Eval: {eval_state}, {eval_status.completed_cases} complete, "
                f"{eval_status.active_cases} active"
            )
        return "  |  ".join(parts)

    def approval_text() -> str:
        request = approval_broker.pending_request()
        if request is None:
            return ""
        summary = " ".join(request.summary.split())
        return f"Approval ({request.risk_level.value}): {summary} [y/N] (/stop)"

    try:
        inline_intro_shown = False
        while True:
            if mode == "fullscreen":
                try:
                    terminal_ui = FullScreenTerminalUI(
                        prompt_label=console.prompt_label,
                        status_text=status_text,
                        approval_text=approval_text,
                        is_busy=is_busy,
                    )
                    terminal_ui.append_output(banner + "\n")
                    with redirect_stdout(terminal_ui.stream):
                        console.interactive_intro()
                    action = terminal_ui.run(process_prompt)
                except Exception as exc:
                    print(f"mikucli: full-screen Terminal UI unavailable; using inline mode: {exc}", file=sys.stderr)
                    mode = "inline"
                    continue
                if action == "exit":
                    return 0
                mode = "inline"
                print("mikucli: switched to inline Terminal UI mode.")
                continue

            if not inline_intro_shown:
                console.interactive_intro()
                inline_intro_shown = True
            prompt_session: PromptSession[str] | None = PromptSession() if live_prompt else None
            with patch_stdout() if prompt_session is not None else nullcontext():
                while mode == "inline":
                    try:
                        prompt = (
                            prompt_session.prompt(console.prompt_label())
                            if prompt_session is not None
                            else input(console.prompt_label())
                        )
                    except KeyboardInterrupt:
                        if not is_busy():
                            continue
                        prompt = "/stop"
                    except EOFError:
                        print()
                        prompt = "/exit"
                    action = process_prompt(prompt)
                    if action == "exit":
                        return 0
                    if action == "fullscreen":
                        break
    finally:
        approval_broker.cancel()
        turn_controller.request_stop()
        turn_controller.wait(timeout=1)
        if eval_controller.is_running():
            print("mikucli: stopping eval suite and waiting for its report...")
        eval_controller.shutdown()
        console.set_approval_handler(None)
        console.set_status_handler(None)
        if mcp_tools is not None:
            mcp_tools.close()

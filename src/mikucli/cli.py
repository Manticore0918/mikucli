from __future__ import annotations

import argparse
import sys
import threading
from pathlib import Path
from typing import Any, Callable

from .codebase.chunking import ChunkingError
from .codebase.embeddings import EmbeddingError
from .codebase.formatting import format_search_results
from .codebase.index import CodebaseIndexError
from .codebase.service import CodebaseService
from .config import Config, load_config
from .console import TerminalConsole
from .evaluation.bench.runner import BenchmarkError, run_benchmarks, summarize_results
from .llm import BigModelClient
from .mcp_config import McpConfigError, load_mcp_config
from .mcp_tools import McpRuntimeError, McpToolSet
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mikucli")
    parser.add_argument("task_prompt", nargs="*", help="Initial task prompt for the agent session.")
    parser.add_argument("--workspace", default=".", help="Workspace directory. Defaults to the current directory.")
    parser.add_argument("--model", default=None, help="GLM model name. Defaults to MIKUCLI_MODEL or glm-5.2.")
    parser.add_argument("--env-file", default=None, help="Path to an additional .env file with high priority.")
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
        print(console.error(exc), file=sys.stderr)
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
    eval_controller = EvalRunController(_eval_runner(client=client, config=config, max_steps=args.max_steps))

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
                    team_mode=team_mode,
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
                    team_mode=team_mode,
                )
                console.print_mode(team_mode=team_mode, mcp_enabled=False, tool_count=len(builtin_tools.schemas()))
                return 0
            if handle_slash_command(
                initial_prompt,
                codebase_service,
                console,
                eval_controller=eval_controller,
                eval_background_allowed=False,
            ):
                return 0
            print(f"{console.prompt_label()}{initial_prompt}")
            result = session.run_turn(initial_prompt)
            console.log_path(result.log_path)
            return 0
        finally:
            if mcp_tools is not None:
                mcp_tools.close()

    console.interactive_intro()
    try:
        while True:
            try:
                prompt = input(console.prompt_label()).strip()
            except EOFError:
                print()
                return 0
            if not prompt:
                continue
            if prompt in {"/exit", "/quit"}:
                return 0
            if prompt == "/mcp":
                if mcp_tools is None:
                    try:
                        mcp_tools = _connect_mcp_tools(config=config, console=console)
                    except (McpConfigError, McpRuntimeError, TimeoutError) as exc:
                        console.print_mcp_enable_error(exc, config.workspace / ".mikucli" / "mcp.json")
                        continue
                    active_tools = mcp_tools
                    mcp_enabled = True
                else:
                    mcp_tools.close()
                    mcp_tools = None
                    active_tools = builtin_tools
                    mcp_enabled = False
                session = _new_session(
                    client=client,
                    config=config,
                    tools=active_tools,
                    console=console,
                    max_steps=args.max_steps,
                    long_term_memory=long_term_memory,
                    team_mode=team_mode,
                )
                console.print_mode(team_mode=team_mode, mcp_enabled=mcp_enabled, tool_count=len(active_tools.schemas()))
                continue
            if prompt == "/team":
                team_mode = not team_mode
                active_tools = mcp_tools if mcp_tools is not None else builtin_tools
                session = _new_session(
                    client=client,
                    config=config,
                    tools=active_tools,
                    console=console,
                    max_steps=args.max_steps,
                    long_term_memory=long_term_memory,
                    team_mode=team_mode,
                )
                console.print_mode(
                    team_mode=team_mode,
                    mcp_enabled=mcp_tools is not None,
                    tool_count=len(active_tools.schemas()),
                )
                continue
            if handle_slash_command(
                prompt,
                codebase_service,
                console,
                eval_controller=eval_controller,
                eval_background_allowed=True,
            ):
                continue
            result = session.run_turn(prompt)
            console.log_path(result.log_path)
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


def _new_session(
    *,
    client: BigModelClient,
    config: Config,
    tools: object,
    console: TerminalConsole,
    max_steps: int,
    long_term_memory: LongTermMemory,
    team_mode: bool,
) -> AgentSession | OrchestratorSession:
    if team_mode:
        return OrchestratorSession(
            client=client,
            model=config.model,
            workspace=config.workspace,
            tools=tools,
            console=console,
            max_steps=max_steps,
            context_window_tokens=config.context_window_tokens,
            long_term_memory=long_term_memory,
        )
    return _new_single_agent_session(
        client=client,
        config=config,
        tools=tools,
        console=console,
        max_steps=max_steps,
        long_term_memory=long_term_memory,
    )


def _connect_mcp_tools(*, config: Config, console: TerminalConsole) -> McpToolSet:
    mcp_config = load_mcp_config(config.workspace)
    mcp_tools = McpToolSet.connect(
        config=mcp_config,
        workspace=config.workspace,
        confirm_tool=console.confirm_tool,
    )
    console.print_mcp_status(mcp_tools.statuses())
    return mcp_tools


CaseFinished = Callable[[Any], None]
EvalRunner = Callable[[Callable[[], bool], CaseFinished | None], tuple[list[Any], Path, Path]]


class EvalRunController:
    def __init__(self, runner: EvalRunner) -> None:
        self.runner = runner
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None
        self.result: tuple[list[Any], Path, Path] | None = None
        self.error: Exception | None = None
        self.lock = threading.Lock()

    def is_running(self) -> bool:
        thread = self.thread
        return thread is not None and thread.is_alive()

    def start(
        self,
        *,
        background: bool,
        on_case_finished: CaseFinished | None = None,
    ) -> tuple[list[Any], Path, Path] | None:
        with self.lock:
            if self.is_running():
                raise RuntimeError("eval suite is already running")
            self.stop_event.clear()
            self.result = None
            self.error = None
            if background:
                self.thread = threading.Thread(
                    target=self._run,
                    args=(on_case_finished,),
                    name="mikucli-eval-suite",
                    daemon=True,
                )
                self.thread.start()
                return None
        if background:
            return None
        self._run(on_case_finished)
        if self.error is not None:
            raise self.error
        return self.result

    def stop(self) -> tuple[list[Any], Path, Path] | None:
        if not self.is_running():
            return None
        self.stop_event.set()
        thread = self.thread
        if thread is not None:
            thread.join()
        if self.error is not None:
            raise self.error
        return self.result

    def _run(self, on_case_finished: CaseFinished | None = None) -> None:
        try:
            self.result = self.runner(self.stop_event.is_set, on_case_finished)
        except Exception as exc:  # pragma: no cover - defensive capture for background threads.
            self.error = exc


def handle_slash_command(
    prompt: str,
    codebase_service: CodebaseService,
    console: TerminalConsole,
    *,
    eval_controller: EvalRunController | None = None,
    eval_background_allowed: bool = False,
) -> bool:
    if prompt == "/lang-chn":
        console.set_language("chn")
        console.language_changed()
        return True

    if prompt == "/lang-eng":
        console.set_language("eng")
        console.language_changed()
        return True

    if prompt == "/index":
        try:
            codebase_service.rebuild_index(progress=console.progress)
        except (ChunkingError, CodebaseIndexError, EmbeddingError, ValueError) as exc:
            print(console.error(exc), file=sys.stderr)
        return True

    if prompt == "/search" or prompt.startswith("/search "):
        query = prompt.removeprefix("/search").strip()
        if not query:
            print(console.search_usage(), file=sys.stderr)
            return True
        try:
            results = codebase_service.search(query, limit=5)
        except (CodebaseIndexError, EmbeddingError) as exc:
            print(console.error(exc), file=sys.stderr)
            return True
        print(format_search_results(results, max_content_chars=1000))
        return True

    if prompt == "/eval" or prompt.startswith("/eval "):
        if prompt not in {"/eval run", "/eval run-back", "/eval stop"}:
            print("mikucli: usage: /eval run | /eval run-back | /eval stop", file=sys.stderr)
            return True
        if eval_controller is None:
            print("mikucli: eval suite is not available in this context.", file=sys.stderr)
            return True
        if prompt == "/eval stop":
            print("mikucli: stopping eval suite after the current benchmark case...")
            try:
                stopped_result = eval_controller.stop()
            except BenchmarkError as exc:
                print(f"mikucli: {exc}", file=sys.stderr)
                return True
            if stopped_result is None:
                print("mikucli: no eval suite is running.")
                return True
            results, result_path, report_path = stopped_result
            _print_eval_summary(results, result_path, report_path, stopped=True)
            return True
        background = prompt == "/eval run-back"
        if background and not eval_background_allowed:
            print("mikucli: /eval run-back is only available in an interactive session.", file=sys.stderr)
            return True
        print("mikucli: starting eval suite...")
        try:
            started_result = eval_controller.start(
                background=background,
                on_case_finished=None if background else _print_eval_case_finished,
            )
        except RuntimeError as exc:
            print(f"mikucli: {exc}", file=sys.stderr)
            return True
        except BenchmarkError as exc:
            print(f"mikucli: {exc}", file=sys.stderr)
            return True
        if started_result is None:
            print("mikucli: eval suite is running in the background. Type /eval stop to stop and write a report.")
            return True
        results, result_path, report_path = started_result
        _print_eval_summary(results, result_path, report_path, stopped=False)
        return True

    return False


def _eval_runner(*, client: BigModelClient, config: Config, max_steps: int) -> EvalRunner:
    def run(
        stop_requested: Callable[[], bool],
        on_case_finished: CaseFinished | None = None,
    ) -> tuple[list[Any], Path, Path]:
        return run_benchmarks(
            root=config.workspace,
            client=client,
            model=config.model,
            max_steps=max_steps,
            context_window_tokens=config.context_window_tokens,
            stop_requested=stop_requested,
            on_case_finished=on_case_finished,
        )

    return run


def _print_eval_case_finished(result: Any) -> None:
    status = "MISSION SUCCEED" if result.passed else "MISSION FAILED"
    metrics = result.metrics
    print(
        f"mikucli: {status}: {result.case_id} "
        f"(total={metrics.elapsed_seconds:.3f}s, "
        f"agent={metrics.agent_latency_seconds:.3f}s, "
        f"llm={metrics.llm_latency_seconds:.3f}s, "
        f"tool_calls={metrics.tool_call_count}, "
        f"model_retries={metrics.model_retries}, "
        f"step_retries={metrics.step_retries})"
    )
    for reason in result.failure_reasons:
        print(f"mikucli:   failure [{reason.category}/{reason.source}]: {reason.message}")


def _print_eval_summary(results: list[Any], result_path: Path, report_path: Path, *, stopped: bool) -> None:
    summary = summarize_results(results, stopped=stopped)
    status = "stopped" if stopped else "complete"
    print(f"mikucli: eval suite {status}: {summary.passed_cases}/{summary.total_cases} benchmark cases passed")
    print(f"mikucli: success rate: {summary.success_rate * 100:.1f}%")
    print(
        "mikucli: "
        f"tool_calls={summary.tool_call_count}, "
        f"model_retries={summary.model_retries}, "
        f"step_retries={summary.step_retries}, "
        f"total_latency={summary.elapsed_seconds:.3f}s, "
        f"agent_latency={summary.agent_latency_seconds:.3f}s, "
        f"llm_latency={summary.llm_latency_seconds:.3f}s"
    )
    print(f"mikucli: benchmark results: {result_path}")
    print(f"mikucli: benchmark report: {report_path}")


if __name__ == "__main__":
    raise SystemExit(main())

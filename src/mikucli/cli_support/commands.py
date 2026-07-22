from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path

from mikucli.codebase.chunking import ChunkingError
from mikucli.codebase.embeddings import EmbeddingError
from mikucli.codebase.formatting import format_search_results
from mikucli.codebase.index import CodebaseIndexError
from mikucli.codebase.service import CodebaseService
from mikucli.console import TerminalConsole
from mikucli.evaluation.bench.runner import BenchmarkError
from mikucli.skills import SkillError, SkillRegistry

from .dashboard import DashboardLaunchError, launch_dashboard, stop_dashboard
from .evaluation import (
    EvalRunController,
    print_eval_case_finished,
    print_eval_case_started,
    print_eval_summary,
)


def handle_slash_command(
    prompt: str,
    codebase_service: CodebaseService,
    console: TerminalConsole,
    *,
    skill_registry: SkillRegistry | None = None,
    eval_controller: EvalRunController | None = None,
    eval_background_allowed: bool = False,
    dashboard_workspace: Path | None = None,
    dashboard_launcher: Callable[[Path], tuple[str, bool]] = launch_dashboard,
    stop_handler: Callable[[], bool] | None = None,
    dashboard_stopper: Callable[[], bool] = stop_dashboard,
) -> bool:
    """Handle Slash Commands that do not change the active session mode."""

    if prompt == "/lang-chn":
        console.set_language("chn")
        console.language_changed()
        return True
    if prompt == "/lang-eng":
        console.set_language("eng")
        console.language_changed()
        return True
    if prompt == "/skills":
        if skill_registry is None:
            print("mikucli: Skill registry is not available in this context.", file=sys.stderr)
            return True
        try:
            console.print_skills(skill_registry.list_entries())
        except SkillError as exc:
            print(console.error(exc.localized(console.language)), file=sys.stderr)
        return True
    if prompt == "/stop":
        if stop_handler is not None and stop_handler():
            print("mikucli: stop requested for the current agent process.")
            return True
        if eval_controller is not None and eval_controller.is_running():
            return _handle_eval_command(
                "/eval stop",
                eval_controller=eval_controller,
                eval_background_allowed=eval_background_allowed,
            )
        if dashboard_stopper():
            print("mikucli: dashboard backend stopped.")
            return True
        print("mikucli: no process is currently running.")
        return True
    if prompt == "/dashboard":
        if dashboard_workspace is None:
            print("mikucli: dashboard is not available in this context.", file=sys.stderr)
            return True
        try:
            url, started = dashboard_launcher(dashboard_workspace)
        except (DashboardLaunchError, OSError) as exc:
            print(f"mikucli: could not launch dashboard: {exc}", file=sys.stderr)
            return True
        if started:
            print(f"mikucli: dashboard backend started at {url} and opened in your default browser.")
        else:
            print(f"mikucli: dashboard already running at {url}; opened in your default browser.")
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
        return _handle_eval_command(
            prompt,
            eval_controller=eval_controller,
            eval_background_allowed=eval_background_allowed,
        )
    return False


def _handle_eval_command(
    prompt: str,
    *,
    eval_controller: EvalRunController | None,
    eval_background_allowed: bool,
) -> bool:
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
        print_eval_summary(results, result_path, report_path, stopped=True)
        return True
    background = prompt == "/eval run-back"
    if background and not eval_background_allowed:
        print("mikucli: /eval run-back is only available in an interactive session.", file=sys.stderr)
        return True
    print("mikucli: starting eval suite...")
    try:
        started_result = eval_controller.start(
            background=background,
            on_case_started=None if background else print_eval_case_started,
            on_case_finished=None if background else print_eval_case_finished,
        )
    except (RuntimeError, BenchmarkError) as exc:
        print(f"mikucli: {exc}", file=sys.stderr)
        return True
    if started_result is None:
        print("mikucli: eval suite is running in the background. Type /eval stop to stop and write a report.")
        return True
    results, result_path, report_path = started_result
    print_eval_summary(results, result_path, report_path, stopped=False)
    return True

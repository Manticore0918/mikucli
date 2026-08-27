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
from mikucli.skills import SkillError, SkillRegistry

from .dashboard import DashboardLaunchError, launch_dashboard, stop_dashboard
from .evaluation import (
    EvalCaseSelection,
    EvalResult,
    EvalRunController,
    print_eval_case_finished,
    print_eval_case_started,
    print_eval_summary,
)


EVAL_USAGE = (
    "mikucli: usage: /eval run | /eval run-single | /eval run-multi | /eval run-stress | "
    "/eval run-stress-s | /eval run-stress-m | /eval run-back | /eval stop"
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
    if prompt not in {
        "/eval run",
        "/eval run-single",
        "/eval run-multi",
        "/eval run-stress",
        "/eval run-stress-s",
        "/eval run-stress-m",
        "/eval run-back",
        "/eval stop",
    }:
        print(EVAL_USAGE, file=sys.stderr)
        return True
    if eval_controller is None:
        print("mikucli: eval suite is not available in this context.", file=sys.stderr)
        return True
    if prompt == "/eval stop":
        if not eval_controller.stop():
            print("mikucli: no eval suite is running.")
            return True
        print(
            "mikucli: eval stop requested; "
            "the completion summary will print when the report is ready."
        )
        return True
    requested_background = prompt == "/eval run-back"
    if requested_background and not eval_background_allowed:
        print("mikucli: /eval run-back is only available in an interactive session.", file=sys.stderr)
        return True
    run_on_worker = requested_background or eval_background_allowed
    case_selection: EvalCaseSelection = "all"
    if prompt == "/eval run-single":
        case_selection = "single"
    elif prompt == "/eval run-multi":
        case_selection = "multi"
    elif prompt == "/eval run-stress":
        case_selection = "stress"
    elif prompt == "/eval run-stress-s":
        case_selection = "stress-single"
    elif prompt == "/eval run-stress-m":
        case_selection = "stress-multi"
    selection_label = {
        "all": "",
        "single": "single-agent ",
        "multi": "multi-agent ",
        "stress": "stress ",
        "stress-single": "single-agent stress ",
        "stress-multi": "multi-agent stress ",
    }[case_selection]
    print(f"mikucli: starting {selection_label}eval suite...")
    try:
        started_result = eval_controller.start(
            background=run_on_worker,
            exclusive=not requested_background,
            on_case_started=None if requested_background else print_eval_case_started,
            on_case_finished=None if requested_background else print_eval_case_finished,
            on_completed=_print_background_eval_completed if run_on_worker else None,
            on_failed=_print_background_eval_failed if run_on_worker else None,
            case_selection=case_selection,
        )
    except Exception as exc:
        print(f"mikucli: {exc}", file=sys.stderr)
        return True
    if started_result is None:
        if requested_background:
            print("mikucli: eval suite is running in the background. Type /eval stop to request a stop.")
        else:
            print("mikucli: eval suite is running. Type /stop or /eval stop to request a stop.")
        return True
    results, result_path, report_path = started_result
    print_eval_summary(results, result_path, report_path, stopped=False)
    return True


def _print_background_eval_completed(
    completed: EvalResult,
    stopped: bool,
) -> None:
    results, result_path, report_path = completed
    print_eval_summary(results, result_path, report_path, stopped=stopped)


def _print_background_eval_failed(exc: Exception) -> None:
    print(f"mikucli: eval suite failed: {exc}", file=sys.stderr)

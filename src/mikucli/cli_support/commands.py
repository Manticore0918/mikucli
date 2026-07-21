from __future__ import annotations

import sys

from mikucli.codebase.chunking import ChunkingError
from mikucli.codebase.embeddings import EmbeddingError
from mikucli.codebase.formatting import format_search_results
from mikucli.codebase.index import CodebaseIndexError
from mikucli.codebase.service import CodebaseService
from mikucli.console import TerminalConsole
from mikucli.evaluation.bench.runner import BenchmarkError

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
    eval_controller: EvalRunController | None = None,
    eval_background_allowed: bool = False,
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

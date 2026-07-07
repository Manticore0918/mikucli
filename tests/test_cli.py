from __future__ import annotations

import io
import unittest
from contextlib import redirect_stdout
from contextlib import redirect_stderr
from pathlib import Path

from mikucli.cli import build_parser, configure_output_encoding, handle_slash_command, render_banner
from mikucli.console import TerminalConsole
from mikucli.evaluation.bench.models import BenchmarkMetrics, BenchmarkResult, CheckResult, EvalCost


class CliTests(unittest.TestCase):
    def test_parser_accepts_task_workspace_and_model(self) -> None:
        args = build_parser().parse_args(["hello", "--workspace", ".", "--model", "glm-5.2"])
        self.assertEqual(args.task_prompt, ["hello"])
        self.assertEqual(args.workspace, ".")
        self.assertEqual(args.model, "glm-5.2")

    def test_parser_accepts_env_file_path(self) -> None:
        args = build_parser().parse_args(["--env-file", "mikucli.env"])
        self.assertEqual(args.env_file, "mikucli.env")

    def test_parser_accepts_context_window_tokens(self) -> None:
        args = build_parser().parse_args(["--context-window-tokens", "64000"])
        self.assertEqual(args.context_window_tokens, 64000)

    def test_banner_renders_boxed_art(self) -> None:
        lines = render_banner().splitlines()
        self.assertTrue(lines[0].startswith("╔"))
        self.assertTrue(lines[-1].startswith("╚"))
        self.assertIn("███╗", render_banner())
        self.assertIn("███████╗", render_banner())
        self.assertEqual(len({len(line) for line in lines}), 1)

    def test_configure_output_encoding_is_safe_to_call(self) -> None:
        configure_output_encoding()

    def test_language_slash_commands_switch_console_language(self) -> None:
        console = TerminalConsole()
        buffer = io.StringIO()

        with redirect_stdout(buffer):
            self.assertTrue(handle_slash_command("/lang-chn", _UnusedCodebaseService(), console))
            self.assertEqual(console.language, "chn")
            self.assertEqual(console.prompt_label(), "你: ")

            self.assertTrue(handle_slash_command("/lang-eng", _UnusedCodebaseService(), console))
            self.assertEqual(console.language, "eng")
            self.assertEqual(console.prompt_label(), "You: ")

        self.assertIn("界面语言已切换为中文", buffer.getvalue())
        self.assertIn("language switched to English", buffer.getvalue())


    def test_eval_run_slash_command_starts_eval_suite(self) -> None:
        console = TerminalConsole()
        buffer = io.StringIO()
        calls = 0

        def eval_runner():
            nonlocal calls
            calls += 1
            result = BenchmarkResult(
                case_id="file_edit:built_in_single_agent",
                task_id="file_edit",
                session_mode="built_in_single_agent",
                passed=True,
                check_results=[CheckResult(name="ok", passed=True)],
                final_answer="done",
                changed_paths=[],
                tool_calls=[],
                approvals=[],
                run_log_path="run-log.json",
                workspace="workspace",
                model="fake-model",
                elapsed_seconds=0.25,
                metrics=BenchmarkMetrics(
                    tool_call_count=2,
                    model_retries=1,
                    step_retries=0,
                    elapsed_seconds=0.25,
                    cost=EvalCost(total_tokens=100),
                ),
            )
            return [result], Path("results.json"), Path("report.md")

        with redirect_stdout(buffer):
            self.assertTrue(handle_slash_command("/eval run", _UnusedCodebaseService(), console, eval_runner=eval_runner))

        self.assertEqual(calls, 1)
        output = buffer.getvalue()
        self.assertIn("mikucli: starting eval suite", output)
        self.assertIn("1/1 benchmark cases passed", output)
        self.assertIn("results.json", output)
        self.assertIn("report.md", output)

    def test_eval_slash_command_prints_usage(self) -> None:
        console = TerminalConsole()
        stdout = io.StringIO()
        stderr = io.StringIO()

        with redirect_stdout(stdout), redirect_stderr(stderr):
            self.assertTrue(handle_slash_command("/eval", _UnusedCodebaseService(), console, eval_runner=lambda: ([], Path(), Path())))

        self.assertEqual(stdout.getvalue(), "")
        self.assertIn("mikucli: usage: /eval run", stderr.getvalue())


class _UnusedCodebaseService:
    pass


if __name__ == "__main__":
    unittest.main()

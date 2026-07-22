from __future__ import annotations

import io
import os
import subprocess
import sys
import tempfile
import time
import unittest
from contextlib import redirect_stdout
from contextlib import redirect_stderr
from pathlib import Path
from threading import Event

from mikucli.cli import EvalRunController, build_parser, configure_output_encoding, handle_slash_command, render_banner
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

        def eval_runner(stop_requested, on_case_started, on_case_finished):
            nonlocal calls
            calls += 1
            if on_case_started is not None:
                on_case_started(_benchmark_case())
            result = _benchmark_result()
            if on_case_finished is not None:
                on_case_finished(result)
            return [result], Path("results.json"), Path("report.md")

        with redirect_stdout(buffer):
            self.assertTrue(
                handle_slash_command(
                    "/eval run",
                    _UnusedCodebaseService(),
                    console,
                    eval_controller=EvalRunController(eval_runner),
                )
            )

        self.assertEqual(calls, 1)
        output = buffer.getvalue()
        self.assertIn("mikucli: starting eval suite", output)
        self.assertIn("mikucli: RUNNING: file_edit:built_in_single_agent", output)
        self.assertIn("mikucli: MISSION SUCCEED: file_edit:built_in_single_agent", output)
        self.assertIn("1/1 benchmark cases passed", output)
        self.assertIn("results.json", output)
        self.assertIn("report.md", output)

    def test_dashboard_slash_command_starts_backend_and_opens_browser(self) -> None:
        console = TerminalConsole()
        buffer = io.StringIO()
        workspace = Path("workspace")
        launched_with: list[Path] = []

        def dashboard_launcher(path: Path) -> tuple[str, bool]:
            launched_with.append(path)
            return "http://127.0.0.1:8765/", True

        with redirect_stdout(buffer):
            self.assertTrue(
                handle_slash_command(
                    "/dashboard",
                    _UnusedCodebaseService(),
                    console,
                    dashboard_workspace=workspace,
                    dashboard_launcher=dashboard_launcher,
                )
            )

        self.assertEqual(launched_with, [workspace])
        self.assertIn("dashboard backend started", buffer.getvalue())
        self.assertIn("http://127.0.0.1:8765/", buffer.getvalue())

    def test_eval_stop_slash_command_stops_background_eval_suite(self) -> None:
        console = TerminalConsole()
        buffer = io.StringIO()
        started = Event()

        def eval_runner(stop_requested, on_case_started, on_case_finished):
            started.set()
            while not stop_requested():
                time.sleep(0.01)
            return [_benchmark_result()], Path("stopped-results.json"), Path("stopped-report.md")

        controller = EvalRunController(eval_runner)

        with redirect_stdout(buffer):
            self.assertTrue(
                handle_slash_command(
                    "/eval run-back",
                    _UnusedCodebaseService(),
                    console,
                    eval_controller=controller,
                    eval_background_allowed=True,
                )
            )
            self.assertTrue(started.wait(timeout=1))
            self.assertTrue(handle_slash_command("/eval stop", _UnusedCodebaseService(), console, eval_controller=controller))

        output = buffer.getvalue()
        self.assertIn("eval suite is running in the background", output)
        self.assertIn("mikucli: stopping eval suite", output)
        self.assertIn("mikucli: eval suite stopped", output)
        self.assertIn("stopped-results.json", output)
        self.assertIn("stopped-report.md", output)

    def test_eval_slash_command_prints_usage(self) -> None:
        console = TerminalConsole()
        stdout = io.StringIO()
        stderr = io.StringIO()

        with redirect_stdout(stdout), redirect_stderr(stderr):
            self.assertTrue(
                handle_slash_command(
                    "/eval",
                    _UnusedCodebaseService(),
                    console,
                    eval_controller=EvalRunController(lambda stop_requested, on_case_started, on_case_finished: ([], Path(), Path())),
                )
            )

        self.assertEqual(stdout.getvalue(), "")
        self.assertIn("mikucli: usage: /eval run | /eval run-back | /eval stop", stderr.getvalue())

    def test_mikucli_subprocess_smoke(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            env_file = root / "mikucli.env"
            env_file.write_text("BIGMODEL_API_KEY=dummy-key\n", encoding="utf-8")
            env = os.environ.copy()
            src_path = str(Path(__file__).resolve().parents[1] / "src")
            env["PYTHONPATH"] = src_path + os.pathsep + env.get("PYTHONPATH", "")

            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "mikucli.cli",
                    "--workspace",
                    str(root),
                    "--env-file",
                    str(env_file),
                    "/lang-eng",
                ],
                text=True,
                encoding="utf-8",
                errors="replace",
                capture_output=True,
                timeout=30,
                check=False,
                env=env,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertIn("language switched to English", completed.stdout)


class _UnusedCodebaseService:
    pass


class _BenchmarkCase:
    id = "file_edit:built_in_single_agent"


def _benchmark_case() -> _BenchmarkCase:
    return _BenchmarkCase()


def _benchmark_result() -> BenchmarkResult:
    return BenchmarkResult(
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


if __name__ == "__main__":
    unittest.main()

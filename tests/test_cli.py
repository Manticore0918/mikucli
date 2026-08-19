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
from types import SimpleNamespace
from unittest.mock import patch

from mikucli.cli import (
    EvalRunController,
    build_parser,
    clear_terminal,
    configure_output_encoding,
    handle_slash_command,
    render_banner,
)
from mikucli.cli_support.evaluation import eval_runner as build_eval_runner
from mikucli.console import TerminalConsole
from mikucli.evaluation.bench.models import BenchmarkMetrics, BenchmarkResult, CheckResult, EvalCost


class CliTests(unittest.TestCase):
    def test_parser_accepts_task_workspace_and_model(self) -> None:
        args = build_parser().parse_args(
            [
                "hello",
                "--workspace",
                ".",
                "--model",
                "provider/model-name",
                "--api-key",
                "provider-key",
                "--base-url",
                "https://provider.test/v1/chat/completions",
            ]
        )
        self.assertEqual(args.task_prompt, ["hello"])
        self.assertEqual(args.workspace, ".")
        self.assertEqual(args.model, "provider/model-name")
        self.assertEqual(args.api_key, "provider-key")
        self.assertEqual(args.base_url, "https://provider.test/v1/chat/completions")

    def test_parser_accepts_env_file_path(self) -> None:
        args = build_parser().parse_args(["--env-file", "mikucli.env"])
        self.assertEqual(args.env_file, "mikucli.env")

    def test_parser_accepts_context_window_tokens(self) -> None:
        args = build_parser().parse_args(["--context-window-tokens", "64000"])
        self.assertEqual(args.context_window_tokens, 64000)

    def test_parser_accepts_langgraph_runtime(self) -> None:
        args = build_parser().parse_args(["--runtime", "langgraph"])
        self.assertEqual(args.runtime, "langgraph")

    def test_banner_renders_boxed_art(self) -> None:
        lines = render_banner().splitlines()
        self.assertTrue(lines[0].startswith("╔"))
        self.assertTrue(lines[-1].startswith("╚"))
        self.assertIn("███╗", render_banner())
        self.assertIn("███████╗", render_banner())
        self.assertEqual(len({len(line) for line in lines}), 1)

    def test_configure_output_encoding_is_safe_to_call(self) -> None:
        configure_output_encoding()

    def test_clear_terminal_uses_cls_for_interactive_windows_terminal(self) -> None:
        with (
            patch("mikucli.cli.sys.stdin.isatty", return_value=True),
            patch("mikucli.cli.sys.stdout.isatty", return_value=True),
            patch("mikucli.cli.os.name", "nt"),
            patch("mikucli.cli.os.system") as system,
        ):
            clear_terminal()

        system.assert_called_once_with("cls")

    def test_clear_terminal_leaves_redirected_output_untouched(self) -> None:
        with (
            patch("mikucli.cli.sys.stdin.isatty", return_value=True),
            patch("mikucli.cli.sys.stdout.isatty", return_value=False),
            patch("mikucli.cli.os.system") as system,
        ):
            clear_terminal()

        system.assert_not_called()

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

        def eval_runner(stop_requested, on_case_started, on_case_finished, case_selection):
            nonlocal calls
            calls += 1
            self.assertEqual(case_selection, "all")
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
        self.assertIn("latency=0.250s", output)
        self.assertNotIn("agent_latency", output)
        self.assertNotIn("llm_latency", output)
        self.assertIn("1/1 benchmark cases passed", output)
        self.assertIn("results.json", output)
        self.assertIn("report.md", output)

    def test_eval_agent_slash_commands_select_matching_cases(self) -> None:
        selections: list[str] = []

        def eval_runner(stop_requested, on_case_started, on_case_finished, case_selection):
            selections.append(case_selection)
            return [_benchmark_result()], Path("results.json"), Path("report.md")

        for command, expected_selection, label in (
            ("/eval run-single", "single", "single-agent"),
            ("/eval run-multi", "multi", "multi-agent"),
        ):
            with self.subTest(command=command):
                output = io.StringIO()
                with redirect_stdout(output):
                    self.assertTrue(
                        handle_slash_command(
                            command,
                            _UnusedCodebaseService(),
                            TerminalConsole(),
                            eval_controller=EvalRunController(eval_runner),
                        )
                    )
                self.assertIn(f"starting {label} eval suite", output.getvalue())
                self.assertEqual(selections[-1], expected_selection)

        self.assertEqual(selections, ["single", "multi"])

    def test_eval_runner_selects_six_cases_for_each_agent_mode(self) -> None:
        config = SimpleNamespace(
            workspace=Path("workspace"),
            model="test-model",
            context_window_tokens=128000,
        )
        runner = build_eval_runner(
            client=object(),
            config=config,
            max_steps=30,
            memory_retriever=object(),
        )

        for selection, suffix in (("single", "_single_agent"), ("multi", "_multi_agent")):
            with self.subTest(selection=selection), patch(
                "mikucli.cli_support.evaluation.run_benchmarks",
                return_value=([], Path("results.json"), Path("report.md")),
            ) as run_benchmarks:
                runner(lambda: False, None, None, selection)

                case_ids = run_benchmarks.call_args.kwargs["case_ids"]
                self.assertEqual(len(case_ids), 6)
                self.assertTrue(all(case_id.endswith(suffix) for case_id in case_ids))

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

    def test_stop_slash_command_stops_active_agent_process_first(self) -> None:
        console = TerminalConsole()
        buffer = io.StringIO()
        calls = 0

        def stop_handler() -> bool:
            nonlocal calls
            calls += 1
            return True

        with redirect_stdout(buffer):
            self.assertTrue(
                handle_slash_command(
                    "/stop",
                    _UnusedCodebaseService(),
                    console,
                    stop_handler=stop_handler,
                    dashboard_stopper=lambda: False,
                )
            )

        self.assertEqual(calls, 1)
        self.assertIn("stop requested for the current agent process", buffer.getvalue())

    def test_eval_stop_slash_command_stops_background_eval_suite(self) -> None:
        console = TerminalConsole()
        buffer = io.StringIO()
        started = Event()

        def eval_runner(stop_requested, on_case_started, on_case_finished, case_selection):
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
            self.assertIsNotNone(controller.wait(timeout=1))

        output = buffer.getvalue()
        self.assertIn("eval suite is running in the background", output)
        self.assertIn("mikucli: eval stop requested", output)
        self.assertIn("mikucli: eval suite stopped", output)
        self.assertIn("stopped-results.json", output)
        self.assertIn("stopped-report.md", output)

    def test_completed_background_eval_reports_and_remains_collectable(self) -> None:
        console = TerminalConsole()
        buffer = io.StringIO()

        def eval_runner(stop_requested, on_case_started, on_case_finished, case_selection):
            return [_benchmark_result()], Path("completed-results.json"), Path("completed-report.md")

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
            completed = controller.wait(timeout=1)
            self.assertIsNotNone(completed)
            self.assertTrue(handle_slash_command("/eval stop", _UnusedCodebaseService(), console, eval_controller=controller))

        output = buffer.getvalue()
        self.assertIn("mikucli: eval suite complete", output)
        self.assertIn("completed-results.json", output)
        self.assertIn("completed-report.md", output)
        self.assertIn("mikucli: no eval suite is running", output)

    def test_background_eval_failure_is_reported_and_collectable(self) -> None:
        console = TerminalConsole()
        stderr = io.StringIO()

        def eval_runner(stop_requested, on_case_started, on_case_finished, case_selection):
            raise OSError("could not write eval report")

        controller = EvalRunController(eval_runner)

        with redirect_stderr(stderr):
            self.assertTrue(
                handle_slash_command(
                    "/eval run-back",
                    _UnusedCodebaseService(),
                    console,
                    eval_controller=controller,
                    eval_background_allowed=True,
                )
            )
            with self.assertRaisesRegex(OSError, "could not write eval report"):
                controller.wait(timeout=1)

        self.assertIn("mikucli: eval suite failed: could not write eval report", stderr.getvalue())

    def test_foreground_eval_failure_is_reported_without_escaping_command(self) -> None:
        console = TerminalConsole()
        stderr = io.StringIO()

        def eval_runner(stop_requested, on_case_started, on_case_finished, case_selection):
            raise OSError("could not create eval workspace")

        with redirect_stderr(stderr):
            self.assertTrue(
                handle_slash_command(
                    "/eval run",
                    _UnusedCodebaseService(),
                    console,
                    eval_controller=EvalRunController(eval_runner),
                )
            )

        self.assertIn("mikucli: could not create eval workspace", stderr.getvalue())

    def test_eval_stop_request_does_not_wait_for_worker(self) -> None:
        started = Event()
        release = Event()

        def eval_runner(stop_requested, on_case_started, on_case_finished, case_selection):
            started.set()
            release.wait(timeout=1)
            return [_benchmark_result()], Path("results.json"), Path("report.md")

        controller = EvalRunController(eval_runner)
        controller.start(background=True)
        self.assertTrue(started.wait(timeout=1))

        self.assertTrue(controller.stop())
        self.assertTrue(controller.is_running())
        release.set()
        self.assertIsNotNone(controller.wait(timeout=1))

    def test_eval_shutdown_stops_and_finalizes_non_daemon_worker(self) -> None:
        started = Event()

        def eval_runner(stop_requested, on_case_started, on_case_finished, case_selection):
            started.set()
            while not stop_requested():
                time.sleep(0.01)
            return [_benchmark_result()], Path("shutdown-results.json"), Path("shutdown-report.md")

        controller = EvalRunController(eval_runner)
        controller.start(background=True)
        self.assertTrue(started.wait(timeout=1))
        self.assertIsNotNone(controller.thread)
        self.assertFalse(controller.thread.daemon)

        controller.shutdown()

        self.assertFalse(controller.is_running())
        self.assertIsNotNone(controller.wait(timeout=0))

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
                    eval_controller=EvalRunController(
                        lambda stop_requested, on_case_started, on_case_finished, case_selection: (
                            [],
                            Path(),
                            Path(),
                        )
                    ),
                )
            )

        self.assertEqual(stdout.getvalue(), "")
        self.assertIn(
            "mikucli: usage: /eval run | /eval run-single | /eval run-multi | "
            "/eval run-back | /eval stop",
            stderr.getvalue(),
        )

    def test_mikucli_subprocess_smoke(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            env_file = root / "mikucli.env"
            env_file.write_text("MIKUCLI_API_KEY=dummy-key\n", encoding="utf-8")
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

    def test_interactive_prompt_is_rendered_and_accepts_exit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            env_file = root / "mikucli.env"
            env_file.write_text("MIKUCLI_API_KEY=dummy-key\n", encoding="utf-8")
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
                ],
                input="/exit\n",
                text=True,
                encoding="utf-8",
                errors="replace",
                capture_output=True,
                timeout=30,
                check=False,
                env=env,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertIn("You:", completed.stdout)


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

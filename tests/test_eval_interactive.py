from __future__ import annotations

import io
import threading
import time
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from mikucli.cli_support.commands import handle_slash_command
from mikucli.cli_support.evaluation import EvalRunController
from mikucli.console import TerminalConsole


class InteractiveEvalTests(unittest.TestCase):
    def test_eval_run_returns_input_control_while_remaining_exclusive(self) -> None:
        started = threading.Event()
        release = threading.Event()

        def runner(stop_requested, on_case_started, on_case_finished, case_selection):
            del stop_requested, on_case_started, on_case_finished, case_selection
            started.set()
            release.wait(timeout=2)
            return [], Path("results.json"), Path("report.md")

        controller = EvalRunController(runner)
        output = io.StringIO()
        started_at = time.monotonic()
        with redirect_stdout(output):
            handled = handle_slash_command(
                "/eval run",
                _UnusedCodebaseService(),
                TerminalConsole(),
                eval_controller=controller,
                eval_background_allowed=True,
            )
            self.assertTrue(started.wait(timeout=1))
            elapsed = time.monotonic() - started_at
            status = controller.status()
            self.assertTrue(status.running)
            self.assertTrue(status.exclusive)
            release.set()
            controller.wait(timeout=1)

        self.assertTrue(handled)
        self.assertLess(elapsed, 0.5)
        self.assertIn("Type /stop or /eval stop", output.getvalue())


class _UnusedCodebaseService:
    pass


if __name__ == "__main__":
    unittest.main()

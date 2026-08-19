from __future__ import annotations

import threading
import time
import sys
import tempfile
import unittest
from pathlib import Path

from mikucli.agent_runtime.contracts import SessionResult
from mikucli.cli_support.evaluation import EvalRunController
from mikucli.cli_support.interactive import ApprovalBroker, InteractiveTurnController
from mikucli.console import TerminalConsole
from mikucli.tools import ToolApprovalRequest, ToolRegistry, ToolRiskLevel
from mikucli.workspace import Workspace


class InteractiveControllerTests(unittest.TestCase):
    def test_stop_reaches_active_turn_and_keeps_input_thread_free(self) -> None:
        session = _BlockingSession()
        controller = InteractiveTurnController(TerminalConsole())

        self.assertTrue(controller.start(session, "work", None))
        self.assertTrue(session.started.wait(timeout=1))
        self.assertTrue(controller.request_stop())
        controller.wait(timeout=1)

        self.assertTrue(session.stop_method_called)
        self.assertFalse(controller.is_running())

    def test_stop_terminates_an_active_shell_process(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            registry = ToolRegistry(Workspace(Path(tmp)))
            result = []
            command = f'"{sys.executable}" -c "import time; time.sleep(10)"'
            worker = threading.Thread(target=lambda: result.append(registry.run_shell(command, "test", 30)))
            worker.start()

            deadline = time.monotonic() + 2
            while not registry._active_processes and time.monotonic() < deadline:
                time.sleep(0.01)
            self.assertTrue(registry.stop_current_process())
            worker.join(timeout=3)

            self.assertFalse(worker.is_alive())
            self.assertFalse(result[0].ok)

    def test_approval_broker_resolves_worker_request_from_input_thread(self) -> None:
        broker = ApprovalBroker()
        decision: list[bool] = []
        request = ToolApprovalRequest(
            tool_name="run_shell",
            risk_level=ToolRiskLevel.HIGH,
            workspace=".",
            summary="test",
        )
        worker = threading.Thread(target=lambda: decision.append(broker.request(request)))
        worker.start()

        deadline = time.monotonic() + 1
        while not broker.has_pending() and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertTrue(broker.resolve(True))
        worker.join(timeout=1)

        self.assertEqual(decision, [True])

    def test_eval_status_tracks_completed_and_active_cases(self) -> None:
        paused = threading.Event()
        release = threading.Event()

        def runner(stop_requested, on_case_started, on_case_finished, case_selection):
            del stop_requested, case_selection
            on_case_started("case-1")
            on_case_finished("case-1")
            on_case_started("case-2")
            paused.set()
            release.wait(timeout=2)
            on_case_finished("case-2")
            return [], Path("results.json"), Path("report.md")

        controller = EvalRunController(runner)
        controller.start(background=True)
        self.assertTrue(paused.wait(timeout=1))

        status = controller.status()
        self.assertTrue(status.running)
        self.assertEqual(status.completed_cases, 1)
        self.assertEqual(status.active_cases, 1)

        release.set()
        controller.wait(timeout=1)
        self.assertFalse(controller.status().running)


class _BlockingSession:
    def __init__(self) -> None:
        self.started = threading.Event()
        self.stop_method_called = False

    def run_turn(self, task_prompt, *, active_skill, stop_requested):
        self.started.set()
        while not stop_requested():
            time.sleep(0.01)
        return SessionResult(final_answer="Stopped by user.", log_path=Path("run-log.json"))

    def request_stop(self) -> None:
        self.stop_method_called = True


if __name__ == "__main__":
    unittest.main()

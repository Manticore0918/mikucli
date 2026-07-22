from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from mikucli.cli_support.dashboard import launch_dashboard, stop_dashboard


class DashboardLauncherTests(unittest.TestCase):
    @patch("mikucli.cli_support.dashboard.webbrowser.open", return_value=True)
    @patch("mikucli.cli_support.dashboard._dashboard_is_ready", side_effect=[False, True])
    @patch("mikucli.cli_support.dashboard.subprocess.Popen")
    def test_launches_backend_with_workspace_store_and_opens_browser(
        self,
        popen: Mock,
        dashboard_is_ready: Mock,
        browser_open: Mock,
    ) -> None:
        process = popen.return_value
        process.poll.return_value = None

        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            url, started = launch_dashboard(workspace)

            command = popen.call_args.args[0]
            self.assertEqual(command[-1], str(workspace / ".mikucli" / "observability"))

        self.assertTrue(started)
        self.assertEqual(url, "http://127.0.0.1:8765/")
        self.assertEqual(command[1:3], ["-m", "mikucli.observability.api"])
        self.assertIs(popen.call_args.kwargs["stdin"], subprocess.DEVNULL)
        browser_open.assert_called_once_with(url)

    @patch("mikucli.cli_support.dashboard.webbrowser.open", return_value=True)
    @patch("mikucli.cli_support.dashboard._dashboard_is_ready", return_value=True)
    @patch("mikucli.cli_support.dashboard.subprocess.Popen")
    def test_reuses_running_backend(self, popen: Mock, dashboard_is_ready: Mock, browser_open: Mock) -> None:
        url, started = launch_dashboard(Path("workspace"))

        self.assertFalse(started)
        popen.assert_not_called()
        browser_open.assert_called_once_with(url)

    @patch("mikucli.cli_support.dashboard._dashboard_process")
    def test_stops_backend_started_by_current_cli(self, process: Mock) -> None:
        process.poll.return_value = None

        self.assertTrue(stop_dashboard())

        process.terminate.assert_called_once_with()
        process.wait.assert_called_once_with(timeout=2)


if __name__ == "__main__":
    unittest.main()

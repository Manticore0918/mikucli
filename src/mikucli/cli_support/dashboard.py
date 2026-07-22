from __future__ import annotations

import os
import subprocess
import sys
import time
import webbrowser
from pathlib import Path
from threading import Lock
from urllib.error import URLError
from urllib.request import urlopen


DASHBOARD_HOST = "127.0.0.1"
DASHBOARD_PORT = 8765
DASHBOARD_START_TIMEOUT_SECONDS = 5.0
_dashboard_process: subprocess.Popen[bytes] | None = None
_dashboard_process_lock = Lock()


class DashboardLaunchError(RuntimeError):
    """Raised when the local observability dashboard cannot be opened."""


def launch_dashboard(workspace: Path) -> tuple[str, bool]:
    """Start the dashboard backend when needed and open it in the default browser."""

    url = f"http://{DASHBOARD_HOST}:{DASHBOARD_PORT}/"
    started = False

    global _dashboard_process
    if not _dashboard_is_ready(url):
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "mikucli.observability.api",
                "--host",
                DASHBOARD_HOST,
                "--port",
                str(DASHBOARD_PORT),
                "--store-root",
                str(workspace / ".mikucli" / "observability"),
            ],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            **_detached_process_options(),
        )
        with _dashboard_process_lock:
            _dashboard_process = process
        started = True
        deadline = time.monotonic() + DASHBOARD_START_TIMEOUT_SECONDS
        while time.monotonic() < deadline:
            if _dashboard_is_ready(url):
                break
            if process.poll() is not None:
                raise DashboardLaunchError(
                    f"dashboard backend exited during startup (exit code {process.returncode}); "
                    f"port {DASHBOARD_PORT} may already be in use"
                )
            time.sleep(0.05)
        else:
            process.terminate()
            with _dashboard_process_lock:
                _dashboard_process = None
            raise DashboardLaunchError(f"dashboard backend did not become ready at {url}")

    try:
        opened = webbrowser.open(url)
    except webbrowser.Error as exc:
        raise DashboardLaunchError(f"could not open the default browser: {exc}") from exc
    if not opened:
        raise DashboardLaunchError(f"could not open the default browser; visit {url} manually")
    return url, started


def stop_dashboard() -> bool:
    """Stop the dashboard backend launched by this mikucli process."""

    global _dashboard_process
    with _dashboard_process_lock:
        process = _dashboard_process
        _dashboard_process = None
    if process is None or process.poll() is not None:
        return False
    process.terminate()
    try:
        process.wait(timeout=2)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=2)
    return True


def _dashboard_is_ready(url: str) -> bool:
    try:
        with urlopen(url, timeout=0.25) as response:
            return response.status == 200 and b"<title>mikucli observability</title>" in response.read(1024)
    except (OSError, URLError):
        return False


def _detached_process_options() -> dict[str, object]:
    if os.name == "nt":
        return {
            "creationflags": subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS,
        }
    return {"start_new_session": True}

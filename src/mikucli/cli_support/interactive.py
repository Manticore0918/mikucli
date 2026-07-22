from __future__ import annotations

import threading
from typing import Any

from mikucli.console import TerminalConsole
from mikucli.skills import Skill
from mikucli.tools import ToolApprovalRequest


class ApprovalBroker:
    """Bridge a tool approval requested by a worker thread to the input thread."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._pending = False
        self._decision = False
        self._resolved = threading.Event()

    def request(self, request: ToolApprovalRequest) -> bool:
        del request
        with self._lock:
            self._pending = True
            self._decision = False
            self._resolved.clear()
        self._resolved.wait()
        with self._lock:
            return self._decision

    def has_pending(self) -> bool:
        with self._lock:
            return self._pending

    def resolve(self, approved: bool) -> bool:
        with self._lock:
            if not self._pending:
                return False
            self._decision = approved
            self._pending = False
            self._resolved.set()
            return True

    def cancel(self) -> bool:
        return self.resolve(False)


class InteractiveTurnController:
    """Run one agent turn off the input thread so `/stop` remains available."""

    def __init__(self, console: TerminalConsole) -> None:
        self.console = console
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._session: Any | None = None

    def is_running(self) -> bool:
        with self._lock:
            thread = self._thread
        return thread is not None and thread.is_alive()

    def start(self, session: Any, task_prompt: str, active_skill: Skill | None) -> bool:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return False
            self._stop_event.clear()
            self._session = session
            self._thread = threading.Thread(
                target=self._run,
                args=(session, task_prompt, active_skill),
                name="mikucli-agent-turn",
                daemon=True,
            )
            self._thread.start()
        return True

    def request_stop(self) -> bool:
        with self._lock:
            thread = self._thread
            session = self._session
            if thread is None or not thread.is_alive():
                return False
            self._stop_event.set()
        request_stop = getattr(session, "request_stop", None)
        if callable(request_stop):
            request_stop()
        return True

    def wait(self, timeout: float | None = None) -> None:
        with self._lock:
            thread = self._thread
        if thread is not None:
            thread.join(timeout)

    def _run(self, session: Any, task_prompt: str, active_skill: Skill | None) -> None:
        try:
            result = session.run_turn(
                task_prompt,
                active_skill=active_skill,
                stop_requested=self._stop_event.is_set,
            )
        except Exception as exc:
            print(self.console.error(exc))
        else:
            self.console.log_path(result.log_path)
        finally:
            with self._lock:
                self._session = None

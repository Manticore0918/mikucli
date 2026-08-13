from __future__ import annotations

import os
import signal
import subprocess
import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, BinaryIO


MAX_CAPTURE_BYTES = 64 * 1024


@dataclass(frozen=True)
class IsolatedProcessResult:
    returncode: int
    stdout: str
    stderr: str
    timed_out: bool = False


@dataclass
class _TailCapture:
    limit: int = MAX_CAPTURE_BYTES
    data: bytearray = field(default_factory=bytearray)
    total_bytes: int = 0

    def append(self, chunk: bytes) -> None:
        self.total_bytes += len(chunk)
        if len(chunk) >= self.limit:
            self.data[:] = chunk[-self.limit :]
            return
        overflow = len(self.data) + len(chunk) - self.limit
        if overflow > 0:
            del self.data[:overflow]
        self.data.extend(chunk)

    def text(self) -> str:
        output = self.data.decode("utf-8", errors="replace")
        if self.total_bytes <= self.limit:
            return output
        omitted = self.total_bytes - len(self.data)
        return f"[... {omitted} output bytes omitted ...]\n{output}"


def run_isolated_python(
    workspace: Path,
    script_arguments: list[str],
    *,
    timeout_seconds: int,
) -> IsolatedProcessResult:
    process_options: dict[str, Any]
    if os.name == "nt":
        process_options = {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
    else:
        process_options = {"start_new_session": True}
    process = subprocess.Popen(
        [sys.executable, "-I", "-B", "-c", *script_arguments],
        cwd=workspace,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=_isolated_environment(),
        **process_options,
    )
    assert process.stdout is not None
    assert process.stderr is not None
    stdout_capture = _TailCapture()
    stderr_capture = _TailCapture()
    readers = [
        threading.Thread(target=_drain_pipe, args=(process.stdout, stdout_capture), daemon=True),
        threading.Thread(target=_drain_pipe, args=(process.stderr, stderr_capture), daemon=True),
    ]
    for reader in readers:
        reader.start()
    timed_out = False
    try:
        process.wait(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        timed_out = True
        _terminate_process_tree(process)
        process.wait()
    for reader in readers:
        reader.join(timeout=5)
    for pipe, reader in zip((process.stdout, process.stderr), readers, strict=True):
        if reader.is_alive():
            pipe.close()
            reader.join(timeout=1)
    return IsolatedProcessResult(
        process.returncode,
        stdout_capture.text(),
        stderr_capture.text(),
        timed_out=timed_out,
    )


def _isolated_environment() -> dict[str, str]:
    allowed = {
        "COMSPEC", "LANG", "LC_ALL", "PATH", "PATHEXT", "SYSTEMROOT",
        "TEMP", "TMP", "TMPDIR", "WINDIR",
    }
    environment = {key: value for key, value in os.environ.items() if key.upper() in allowed}
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["PYTHONIOENCODING"] = "utf-8"
    return environment


def _terminate_process_tree(process: subprocess.Popen[Any]) -> None:
    if process.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/PID", str(process.pid), "/T", "/F"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        if process.poll() is None:
            process.kill()
        return
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def _drain_pipe(pipe: BinaryIO, capture: _TailCapture) -> None:
    try:
        while chunk := pipe.read(8192):
            capture.append(chunk)
    except (OSError, ValueError):
        pass

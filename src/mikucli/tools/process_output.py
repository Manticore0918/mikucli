from __future__ import annotations

import locale
import subprocess
import time
from collections.abc import Callable
from threading import Event, Lock, Thread
from typing import BinaryIO

from mikucli.run_control import check_active, current_budget


def collect_process_output(
    process: subprocess.Popen[bytes],
    terminate: Callable,
    timeout: float,
) -> tuple[str, str, str]:
    """Drain both pipes concurrently while bounding retained and generated output."""
    budget = current_budget()
    limit = (
        max(0, budget.limits.max_output_bytes - budget.output_bytes)
        if budget
        else 1_048_576
    )
    lock, exceeded = Lock(), Event()
    chunks: list[list[bytes]] = [[], []]
    received = 0

    def drain(stream: BinaryIO, index: int) -> None:
        nonlocal received
        try:
            while True:
                chunk = stream.read1(4096)
                if not chunk:
                    return
                with lock:
                    remaining = max(0, limit - received)
                    chunks[index].append(chunk[:remaining])
                    received += len(chunk)
                    if received > limit:
                        exceeded.set()
                        return
        finally:
            stream.close()

    readers = [
        Thread(target=drain, args=(stream, index), daemon=True)
        for index, stream in enumerate((process.stdout, process.stderr))
    ]
    for reader in readers:
        reader.start()
    deadline = time.monotonic() + timeout
    reason = ""
    try:
        while process.poll() is None or any(reader.is_alive() for reader in readers):
            check_active()
            if exceeded.is_set():
                reason = "command output exceeded the configured output limit"
                if budget:
                    budget.fail("output_limit", reason)
                break
            if time.monotonic() >= deadline:
                reason = f"command timed out after {timeout:g} seconds"
                break
            time.sleep(0.02)
        if exceeded.is_set() and not reason:
            reason = "command output exceeded the configured output limit"
            if budget:
                budget.fail("output_limit", reason)
    finally:
        if process.poll() is None or any(reader.is_alive() for reader in readers):
            terminate(process)
        process.wait(timeout=3)
        for reader in readers:
            reader.join(timeout=1)
    encoding = locale.getpreferredencoding(False)
    stdout, stderr = (
        b"".join(parts).decode(encoding, errors="replace") for parts in chunks
    )
    return stdout, stderr, reason

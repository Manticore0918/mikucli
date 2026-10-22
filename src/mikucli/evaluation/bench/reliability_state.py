"""Persistence and queued-MCP probes shared by the offline eval runner."""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from queue import Queue

from mikucli.mcp_runtime.server import ServerRuntime
from mikucli.memory import LongTermMemory
from mikucli.memory.recovery import restore_memory
from mikucli.memory.storage import restore_marker
from mikucli.run_control import RunLimitExceeded, current_budget


def arm_deadline() -> None:
    """Inject elapsed budget time only after the dependency under test starts.

    Startup and trace-store scheduling must not consume a tiny test deadline
    before the HTTP request or MCP command has actually become in-flight.
    The real budget check still detects expiration and performs cancellation.
    """
    budget = current_budget()
    if budget is None:
        raise RuntimeError("deadline probe requires an active task budget")
    budget.started = time.monotonic() - budget.limits.max_duration_seconds + 0.075


def exercise_memory(name: str, workspace: Path) -> list[str]:
    path = workspace / ".mikucli" / "contract-memory.sqlite3"
    memory = LongTermMemory(path)
    memory.save("committed fact")
    if name == "durable_restart_recovery":
        backup = memory.backup()
        path.write_bytes(b"interrupted/corrupt primary")
        restore_marker(path).write_text("interrupted restore", encoding="utf-8")
        restore_memory(path, backup, resume=True)
        actual = [record.content for record in LongTermMemory(path).records]
        return [] if actual == ["committed fact"] else [f"recovery returned {actual!r}"]
    if name == "durable_concurrent_writers":

        def save(index):
            writer = LongTermMemory(path)
            writer.save_many(["shared fact", f"writer {index}"])

        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(save, range(8)))
        actual = {record.content for record in LongTermMemory(path).records}
        expected = {
            "committed fact",
            "shared fact",
            *(f"writer {index}" for index in range(8)),
        }
        return (
            []
            if actual == expected
            else ["concurrent writes lost facts or failed deduplication"]
        )
    try:
        with memory._store.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                "INSERT INTO memories VALUES ('partial', 'partial', 'timestamp')"
            )
            raise InterruptedError("fixture interrupted before commit")
    except InterruptedError:
        pass
    actual = [record.content for record in LongTermMemory(path).records]
    return (
        []
        if actual == ["committed fact"]
        else ["an interrupted transaction became visible"]
    )


class DeadlineQueue(Queue):
    def put(self, item, *args, **kwargs):
        super().put(item, *args, **kwargs)
        arm_deadline()  # The command is definitely queued before time advances.


class QueuedMcpProbe:
    """Drive the real queue submission/guard without a remote server or network."""

    def __init__(self):
        self.runtime = object.__new__(ServerRuntime)
        self.runtime._closed = False
        self.runtime._commands = DeadlineQueue()
        self.cancelled = self.closed = False
        self.runtime._cancel_main_task = self.cancel
        self.runtime.close = self.close

    def cancel(self):
        self.cancelled = True

    def close(self):
        self.closed = True

    def schemas(self):
        return [
            {
                "type": "function",
                "function": {
                    "name": "remote_tool",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]

    def read_only_tool_names(self):
        return {"remote_tool"}

    def requires_approval(self, *args):
        return False

    def invoke(self, *args):
        return self.runtime._submit("call_tool")

    def verify(self):
        if not self.cancelled or not self.closed:
            return ["MCP cancellation did not close the runtime"]
        command = self.runtime._commands.get_nowait()
        try:
            command.guard()
        except RunLimitExceeded:
            return []
        return ["a queued MCP command remained executable after cancellation"]

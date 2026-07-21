from __future__ import annotations

import queue
import threading
from pathlib import Path
from typing import Any

from ..mcp_config import McpConfig
from .server import ServerRuntime
from .types import McpRuntimeError, McpServerStatus


class ThreadedMcpClient:
    """Manage one isolated ServerRuntime per configured MCP server."""

    def __init__(self, config: McpConfig, workspace: Path) -> None:
        self.config = config
        self.workspace = workspace
        self._runtimes: dict[str, ServerRuntime] = {}
        self._closed = False
        created_runtimes: dict[str, ServerRuntime] = {}
        created_lock = threading.Lock()

        def connect_runtime(server_name: str) -> tuple[str, ServerRuntime]:
            name, runtime = self._connect_runtime(server_name)
            with created_lock:
                created_runtimes[name] = runtime
            return name, runtime

        try:
            self._runtimes = {
                name: runtime
                for name, runtime in run_in_daemon_thread_pool(
                    list(config.servers),
                    connect_runtime,
                    max_workers=8,
                )
            }
        except Exception:
            for runtime in reversed(list(created_runtimes.values())):
                runtime.close()
            self.close()
            raise

    def list_tools(self, server_name: str) -> list[Any]:
        return self._runtime(server_name).list_tools()

    def call_tool(self, server_name: str, tool_name: str, arguments: dict[str, Any]) -> Any:
        return self._runtime(server_name).call_tool(tool_name, arguments)

    def statuses(self) -> list[McpServerStatus]:
        statuses: list[McpServerStatus] = []
        for name in self.config.servers:
            runtime = self._runtimes.get(name)
            if runtime is None:
                statuses.append(McpServerStatus(name=name, initialized=False, active=False, error="not connected"))
                continue
            statuses.append(runtime.status())
        return statuses

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for runtime in reversed(list(self._runtimes.values())):
            runtime.close()
        self._runtimes.clear()

    def _connect_runtime(self, server_name: str) -> tuple[str, ServerRuntime]:
        return server_name, ServerRuntime(self.config, self.workspace, server_name)

    def _runtime(self, server_name: str) -> ServerRuntime:
        try:
            return self._runtimes[server_name]
        except KeyError as exc:
            raise McpRuntimeError(f"MCP server is not connected: {server_name}") from exc


def run_in_daemon_thread_pool(items: list[Any], worker: Any, *, max_workers: int) -> list[Any]:
    if max_workers <= 0:
        raise ValueError("max_workers must be positive")
    if not items:
        return []
    work_queue: queue.Queue[tuple[int, Any] | None] = queue.Queue()
    results: list[Any] = [None] * len(items)
    errors: list[BaseException] = []
    error_lock = threading.Lock()
    for index, item in enumerate(items):
        work_queue.put((index, item))
    worker_count = min(max_workers, len(items))
    for _ in range(worker_count):
        work_queue.put(None)

    def run_worker() -> None:
        while True:
            queued = work_queue.get()
            try:
                if queued is None:
                    return
                index, item = queued
                try:
                    results[index] = worker(item)
                except BaseException as exc:
                    with error_lock:
                        errors.append(exc)
            finally:
                work_queue.task_done()

    threads = [
        threading.Thread(target=run_worker, name=f"mikucli-mcp-start-{index + 1}", daemon=True)
        for index in range(worker_count)
    ]
    for thread in threads:
        thread.start()
    work_queue.join()
    for thread in threads:
        thread.join()
    if errors:
        raise errors[0]
    return results

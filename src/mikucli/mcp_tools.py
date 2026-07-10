from __future__ import annotations

import asyncio
import inspect
import queue
import threading
from contextlib import AsyncExitStack
from concurrent.futures import Future, TimeoutError as FutureTimeoutError
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from .mcp_config import McpConfig, McpConfigError, McpToolBinding
from .tools import ConfirmTool, ToolApprovalRequest, ToolResult, ToolRiskLevel


class McpRuntimeError(ValueError):
    pass


@dataclass(frozen=True)
class McpServerStatus:
    name: str
    initialized: bool
    active: bool
    error: str = ""


class McpClient(Protocol):
    def list_tools(self, server_name: str) -> list[Any]: ...
    def call_tool(self, server_name: str, tool_name: str, arguments: dict[str, Any]) -> Any: ...
    def statuses(self) -> list[McpServerStatus]: ...
    def close(self) -> None: ...


@dataclass
class _ConnectedServer:
    name: str
    session: Any
    exit_stack: AsyncExitStack
    tools: list[Any]
    initialized: bool = True
    error: str = ""


@dataclass
class _ServerCommand:
    kind: str
    response: Future[Any]
    tool_name: str = ""
    arguments: dict[str, Any] | None = None
    timeout: int = 30


class _ServerRuntime:
    def __init__(self, config: McpConfig, workspace: Path, server_name: str) -> None:
        self.config = config
        self.workspace = workspace
        self.server_name = server_name
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run_loop, name=f"mikucli-mcp-{server_name}", daemon=True)
        self._commands: queue.Queue[_ServerCommand] = queue.Queue()
        self._ready: Future[_ConnectedServer] = Future()
        self._main_task: asyncio.Task[Any] | None = None
        self._server: _ConnectedServer | None = None
        self._closed = False
        self._thread.start()
        try:
            self._server = self._ready.result(timeout=30)
        except FutureTimeoutError as exc:
            self.close()
            raise McpRuntimeError("MCP operation timed out") from exc
        except Exception:
            self.close()
            raise

    def list_tools(self) -> list[Any]:
        server = self._connected_server()
        return list(server.tools)

    def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        self._connected_server()
        return self._submit("call_tool", tool_name=tool_name, arguments=arguments)

    def status(self) -> McpServerStatus:
        server = self._server
        if server is None:
            return McpServerStatus(name=self.server_name, initialized=False, active=False, error="not connected")
        try:
            server.tools = self._submit("list_tools")
            return McpServerStatus(name=self.server_name, initialized=server.initialized, active=True)
        except Exception as exc:
            return McpServerStatus(
                name=self.server_name,
                initialized=server.initialized,
                active=False,
                error=str(exc),
            )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._thread.is_alive():
            if self._ready.done() and self._server is not None:
                response: Future[Any] = Future()
                self._commands.put(_ServerCommand(kind="close", response=response, timeout=10))
                try:
                    response.result(timeout=10)
                except Exception:
                    pass
            else:
                self._cancel_main_task()
            self._thread.join(timeout=10)

    async def _connect_server(self) -> _ConnectedServer:
        try:
            from mcp import ClientSession, StdioServerParameters
            from mcp.client.stdio import stdio_client
        except ImportError as exc:
            raise McpRuntimeError("official MCP Python SDK is not installed; install dependency 'mcp>=1.27,<2'") from exc

        server = self.config.servers[self.server_name]
        parameters = _stdio_parameters(
            StdioServerParameters,
            command=server.command,
            args=server.args,
            env=server.env or None,
            cwd=str(self.workspace),
        )
        exit_stack = AsyncExitStack()
        try:
            read_stream, write_stream = await exit_stack.enter_async_context(stdio_client(parameters))
            session = await exit_stack.enter_async_context(ClientSession(read_stream, write_stream))
            await session.initialize()
            tools_result = await session.list_tools()
            return _ConnectedServer(
                name=self.server_name,
                session=session,
                exit_stack=exit_stack,
                tools=list(tools_result.tools),
            )
        except Exception:
            await exit_stack.aclose()
            raise

    def _connected_server(self) -> _ConnectedServer:
        if self._server is None:
            raise McpRuntimeError(f"MCP server is not connected: {self.server_name}")
        return self._server

    def _run_loop(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._main_task = self._loop.create_task(self._serve())
        try:
            self._loop.run_until_complete(self._main_task)
        except BaseException as exc:
            if not self._ready.done():
                self._ready.set_exception(exc)
        finally:
            pending = [task for task in asyncio.all_tasks(self._loop) if not task.done()]
            for task in pending:
                task.cancel()
            if pending:
                self._loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            self._loop.close()

    async def _serve(self) -> None:
        close_response: Future[Any] | None = None
        connected: _ConnectedServer | None = None
        try:
            connected = await self._connect_server()
            self._server = connected
            self._ready.set_result(connected)
            while True:
                command = await asyncio.to_thread(self._commands.get)
                if command.kind == "close":
                    close_response = command.response
                    break
                await self._handle_command(command, connected)
        except BaseException as exc:
            if not self._ready.done():
                self._ready.set_exception(exc)
            if close_response is not None and not close_response.done():
                close_response.set_exception(exc)
            raise
        finally:
            cleanup_error: BaseException | None = None
            if connected is not None:
                try:
                    await connected.exit_stack.aclose()
                except BaseException as exc:
                    cleanup_error = exc
            self._server = None
            if close_response is not None and not close_response.done():
                if cleanup_error is None:
                    close_response.set_result(None)
                else:
                    close_response.set_exception(cleanup_error)

    async def _handle_command(self, command: _ServerCommand, server: _ConnectedServer) -> None:
        try:
            if command.kind == "list_tools":
                tools_result = await asyncio.wait_for(server.session.list_tools(), timeout=command.timeout)
                server.tools = list(tools_result.tools)
                command.response.set_result(list(server.tools))
                return
            if command.kind == "call_tool":
                result = await asyncio.wait_for(
                    server.session.call_tool(command.tool_name, command.arguments or {}),
                    timeout=command.timeout,
                )
                command.response.set_result(result)
                return
            command.response.set_exception(McpRuntimeError(f"unknown MCP runtime command: {command.kind}"))
        except asyncio.TimeoutError as exc:
            error = McpRuntimeError("MCP operation timed out")
            error.__cause__ = exc
            command.response.set_exception(error)
        except Exception as exc:
            command.response.set_exception(exc)

    def _submit(
        self,
        kind: str,
        *,
        tool_name: str = "",
        arguments: dict[str, Any] | None = None,
        timeout: int = 30,
    ) -> Any:
        if self._closed:
            raise McpRuntimeError("MCP client is closed")
        response: Future[Any] = Future()
        self._commands.put(
            _ServerCommand(
                kind=kind,
                response=response,
                tool_name=tool_name,
                arguments=arguments,
                timeout=timeout,
            )
        )
        try:
            return response.result(timeout=timeout + 2)
        except FutureTimeoutError as exc:
            raise McpRuntimeError("MCP operation timed out") from exc

    def _cancel_main_task(self) -> None:
        task = self._main_task
        if task is None:
            return
        self._loop.call_soon_threadsafe(task.cancel)


class ThreadedMcpClient:
    def __init__(self, config: McpConfig, workspace: Path) -> None:
        self.config = config
        self.workspace = workspace
        self._runtimes: dict[str, _ServerRuntime] = {}
        self._closed = False
        created_runtimes: dict[str, _ServerRuntime] = {}
        created_lock = threading.Lock()

        def connect_runtime(server_name: str) -> tuple[str, _ServerRuntime]:
            name, runtime = self._connect_runtime(server_name)
            with created_lock:
                created_runtimes[name] = runtime
            return name, runtime

        try:
            self._runtimes = {
                name: runtime
                for name, runtime in _run_in_daemon_thread_pool(
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

    def _connect_runtime(self, server_name: str) -> tuple[str, _ServerRuntime]:
        return server_name, _ServerRuntime(self.config, self.workspace, server_name)

    def _runtime(self, server_name: str) -> _ServerRuntime:
        try:
            return self._runtimes[server_name]
        except KeyError as exc:
            raise McpRuntimeError(f"MCP server is not connected: {server_name}") from exc


def _run_in_daemon_thread_pool(
    items: list[Any],
    worker: Any,
    *,
    max_workers: int,
) -> list[Any]:
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


class McpToolSet:
    def __init__(
        self,
        *,
        config: McpConfig,
        client: McpClient,
        workspace: Path,
        confirm_tool: ConfirmTool | None = None,
    ) -> None:
        self.config = config
        self.client = client
        self.workspace = workspace
        self.confirm_tool = confirm_tool
        self._tool_by_binding = self._load_tools_by_binding()
        self._schemas = self._build_schemas()

    @classmethod
    def connect(
        cls,
        *,
        config: McpConfig,
        workspace: Path,
        confirm_tool: ConfirmTool | None = None,
    ) -> McpToolSet:
        client = ThreadedMcpClient(config, workspace)
        try:
            return cls(config=config, client=client, workspace=workspace, confirm_tool=confirm_tool)
        except Exception:
            client.close()
            raise

    def schemas(self) -> list[dict[str, Any]]:
        return list(self._schemas)

    def read_only_tool_names(self) -> set[str]:
        return {name for name, binding in self.config.tools.items() if binding.read_only}

    def requires_approval(self, name: str) -> bool:
        binding = self.config.tools.get(name)
        return binding is None or binding.risk != ToolRiskLevel.LOW

    def invoke(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        binding = self.config.tools.get(name)
        if binding is None:
            return ToolResult(ok=False, content=f"unknown MCP tool: {name}")
        approval = ToolApprovalRequest(
            tool_name=name,
            risk_level=binding.risk,
            workspace=str(self.workspace),
            summary=f"Run MCP tool: {name}",
            details=(
                f"server: {binding.server}\n"
                f"mcp_tool_name: {binding.mcp_tool_name}\n"
                f"arguments: {arguments}"
            ),
        )
        if not self._approve(approval):
            return ToolResult(ok=False, content="MCP tool call denied by user.")
        try:
            result = self.client.call_tool(binding.server, binding.mcp_tool_name, arguments)
        except Exception as exc:
            return ToolResult(ok=False, content=f"MCP tool call failed: {exc}")
        return _mcp_result_to_tool_result(result)

    def statuses(self) -> list[McpServerStatus]:
        return self.client.statuses()

    def close(self) -> None:
        self.client.close()

    def _load_tools_by_binding(self) -> dict[str, Any]:
        tools_by_binding: dict[str, Any] = {}
        tools_by_server = {server_name: self.client.list_tools(server_name) for server_name in self.config.servers}
        for binding in self.config.tools.values():
            matched = _find_tool(tools_by_server[binding.server], binding.mcp_tool_name)
            if matched is None:
                raise McpConfigError(
                    f"MCP tool binding '{binding.model_name}' references missing tool "
                    f"'{binding.mcp_tool_name}' on server '{binding.server}'"
                )
            tools_by_binding[binding.model_name] = matched
        return tools_by_binding

    def _build_schemas(self) -> list[dict[str, Any]]:
        schemas: list[dict[str, Any]] = []
        for binding in self.config.tools.values():
            tool = self._tool_by_binding[binding.model_name]
            schemas.append(
                {
                    "type": "function",
                    "function": {
                        "name": binding.model_name,
                        "description": _field(tool, "description") or f"MCP tool {binding.model_name}.",
                        "parameters": _field(tool, "inputSchema") or _field(tool, "input_schema") or {"type": "object"},
                    },
                }
            )
        return schemas

    def _approve(self, request: ToolApprovalRequest) -> bool:
        if request.risk_level == ToolRiskLevel.LOW:
            return True
        if self.confirm_tool is None:
            return False
        return self.confirm_tool(request)


def _find_tool(tools: list[Any], name: str) -> Any | None:
    for tool in tools:
        if _field(tool, "name") == name:
            return tool
    return None


def _field(item: Any, name: str) -> Any:
    if isinstance(item, dict):
        return item.get(name)
    return getattr(item, name, None)


def _stdio_parameters(parameters_type: Any, **values: Any) -> Any:
    signature = inspect.signature(parameters_type)
    filtered = {key: value for key, value in values.items() if key in signature.parameters}
    return parameters_type(**filtered)


def _mcp_result_to_tool_result(result: Any) -> ToolResult:
    is_error = bool(_field(result, "isError") or _field(result, "is_error"))
    content = _format_mcp_content(_field(result, "content") or [])
    return ToolResult(ok=not is_error, content=content or ("MCP tool returned an error." if is_error else ""))


def _format_mcp_content(blocks: Any) -> str:
    if not isinstance(blocks, list):
        return str(blocks)
    formatted: list[str] = []
    for block in blocks:
        block_type = _field(block, "type") or type(block).__name__
        if block_type == "text" or _field(block, "text") is not None:
            formatted.append(str(_field(block, "text") or ""))
            continue
        metadata = _content_metadata(block)
        if metadata:
            formatted.append(f"[MCP {block_type} content: {metadata}]")
        else:
            formatted.append(f"[MCP {block_type} content]")
    return "\n".join(part for part in formatted if part)


def _content_metadata(block: Any) -> str:
    parts: list[str] = []
    for name in ("mimeType", "mime_type", "uri", "name"):
        value = _field(block, name)
        if value:
            parts.append(f"{name}={value}")
    return ", ".join(parts)

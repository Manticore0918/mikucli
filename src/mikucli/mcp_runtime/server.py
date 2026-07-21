from __future__ import annotations

import asyncio
import inspect
import queue
import threading
from contextlib import AsyncExitStack
from concurrent.futures import Future, TimeoutError as FutureTimeoutError
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..mcp_config import McpConfig
from .types import McpRuntimeError, McpServerStatus


@dataclass
class ConnectedServer:
    name: str
    session: Any
    exit_stack: AsyncExitStack
    tools: list[Any]
    initialized: bool = True
    error: str = ""


@dataclass
class ServerCommand:
    kind: str
    response: Future[Any]
    tool_name: str = ""
    arguments: dict[str, Any] | None = None
    timeout: int = 30


class ServerRuntime:
    """Own one MCP server's event loop and serialized command queue."""

    def __init__(self, config: McpConfig, workspace: Path, server_name: str) -> None:
        self.config = config
        self.workspace = workspace
        self.server_name = server_name
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run_loop, name=f"mikucli-mcp-{server_name}", daemon=True)
        self._commands: queue.Queue[ServerCommand] = queue.Queue()
        self._ready: Future[ConnectedServer] = Future()
        self._main_task: asyncio.Task[Any] | None = None
        self._server: ConnectedServer | None = None
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
        return list(self._connected_server().tools)

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
            return McpServerStatus(name=self.server_name, initialized=server.initialized, active=False, error=str(exc))

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._thread.is_alive():
            if self._ready.done() and self._server is not None:
                response: Future[Any] = Future()
                self._commands.put(ServerCommand(kind="close", response=response, timeout=10))
                try:
                    response.result(timeout=10)
                except Exception:
                    pass
            else:
                self._cancel_main_task()
            self._thread.join(timeout=10)

    async def _connect_server(self) -> ConnectedServer:
        try:
            from mcp import ClientSession, StdioServerParameters
            from mcp.client.stdio import stdio_client
        except ImportError as exc:
            raise McpRuntimeError(
                "official MCP Python SDK is not installed; install dependency 'mcp>=1.27,<2'"
            ) from exc
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
            return ConnectedServer(
                name=self.server_name,
                session=session,
                exit_stack=exit_stack,
                tools=list(tools_result.tools),
            )
        except Exception:
            await exit_stack.aclose()
            raise

    def _connected_server(self) -> ConnectedServer:
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
        connected: ConnectedServer | None = None
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

    async def _handle_command(self, command: ServerCommand, server: ConnectedServer) -> None:
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
            ServerCommand(
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
        if task is not None:
            self._loop.call_soon_threadsafe(task.cancel)


def _stdio_parameters(parameters_type: Any, **values: Any) -> Any:
    signature = inspect.signature(parameters_type)
    filtered = {key: value for key, value in values.items() if key in signature.parameters}
    return parameters_type(**filtered)

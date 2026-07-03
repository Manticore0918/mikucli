from __future__ import annotations

import asyncio
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from mikucli.mcp_config import McpConfig, McpConfigError, McpServerConfig, McpToolBinding
from mikucli.mcp_tools import (
    McpServerStatus,
    McpToolSet,
    _ConnectedServer,
    _ServerRuntime,
    _run_in_daemon_thread_pool,
)
from mikucli.tools import ToolApprovalRequest, ToolRiskLevel


class FakeMcpClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, dict[str, Any]]] = []
        self.closed = False

    def list_tools(self, server_name: str) -> list[Any]:
        if server_name == "zread":
            return [
                SimpleNamespace(
                    name="read_file",
                    description="Read a file from GitHub.",
                    inputSchema={
                        "type": "object",
                        "required": ["path"],
                        "properties": {"path": {"type": "string"}},
                    },
                ),
                SimpleNamespace(
                    name="write_file",
                    description="Write a file to GitHub.",
                    inputSchema={"type": "object"},
                )
            ]
        return []

    def call_tool(self, server_name: str, tool_name: str, arguments: dict[str, Any]) -> Any:
        self.calls.append((server_name, tool_name, arguments))
        return SimpleNamespace(
            isError=False,
            content=[
                SimpleNamespace(type="text", text="hello from github"),
                SimpleNamespace(type="image", mimeType="image/png", name="preview"),
            ],
        )

    def statuses(self) -> list[McpServerStatus]:
        return [McpServerStatus(name="zread", initialized=True, active=True)]

    def close(self) -> None:
        self.closed = True


class McpToolSetTests(unittest.TestCase):
    def test_builds_model_facing_schema_from_mcp_tool(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tools = McpToolSet(
                config=_config(ToolRiskLevel.LOW),
                client=FakeMcpClient(),
                workspace=Path(tmp),
            )

            schema = tools.schemas()[0]["function"]

            self.assertEqual(schema["name"], "read_github_file")
            self.assertEqual(schema["description"], "Read a file from GitHub.")
            self.assertEqual(schema["parameters"]["required"], ["path"])

    def test_invokes_binding_route_and_formats_result(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            client = FakeMcpClient()
            tools = McpToolSet(
                config=_config(ToolRiskLevel.LOW),
                client=client,
                workspace=Path(tmp),
            )

            result = tools.invoke("read_github_file", {"path": "README.md"})

            self.assertTrue(result.ok)
            self.assertEqual(client.calls, [("zread", "read_file", {"path": "README.md"})])
            self.assertIn("hello from github", result.content)
            self.assertIn("MCP image content", result.content)
            self.assertIn("mimeType=image/png", result.content)

    def test_approval_blocks_medium_risk_binding(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            approvals: list[ToolApprovalRequest] = []
            client = FakeMcpClient()
            tools = McpToolSet(
                config=_config(ToolRiskLevel.MEDIUM),
                client=client,
                workspace=Path(tmp),
                confirm_tool=lambda request: approvals.append(request) or False,
            )

            result = tools.invoke("read_github_file", {"path": "README.md"})

            self.assertFalse(result.ok)
            self.assertEqual(result.content, "MCP tool call denied by user.")
            self.assertEqual(client.calls, [])
            self.assertEqual(approvals[0].tool_name, "read_github_file")
            self.assertEqual(approvals[0].risk_level, ToolRiskLevel.MEDIUM)
            self.assertIn("server: zread", approvals[0].details)

    def test_read_only_tool_names_come_from_binding_flag(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tools = McpToolSet(
                config=_config(ToolRiskLevel.LOW),
                client=FakeMcpClient(),
                workspace=Path(tmp),
            )

            self.assertEqual(tools.read_only_tool_names(), {"read_github_file"})

    def test_rejects_binding_when_server_does_not_offer_tool(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = McpConfig(
                servers={"zread": McpServerConfig(name="zread", command="zread-mcp")},
                tools={
                    "read_github_file": McpToolBinding(
                        model_name="read_github_file",
                        server="zread",
                        mcp_tool_name="missing",
                    )
                },
            )

            with self.assertRaisesRegex(McpConfigError, "missing tool"):
                McpToolSet(config=config, client=FakeMcpClient(), workspace=Path(tmp))

    def test_daemon_thread_pool_caps_parallel_startup_at_eight_workers(self) -> None:
        active = 0
        max_active = 0
        daemon_values: list[bool] = []
        lock = threading.Lock()

        def worker(item: int) -> int:
            nonlocal active, max_active
            with lock:
                active += 1
                max_active = max(max_active, active)
                daemon_values.append(threading.current_thread().daemon)
            time.sleep(0.01)
            with lock:
                active -= 1
            return item * 2

        results = _run_in_daemon_thread_pool(list(range(20)), worker, max_workers=8)

        self.assertEqual(results, [item * 2 for item in range(20)])
        self.assertLessEqual(max_active, 8)
        self.assertTrue(daemon_values)
        self.assertTrue(all(daemon_values))

    def test_server_runtime_closes_context_from_opening_task(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runtime = SameTaskRuntime(_config(ToolRiskLevel.LOW), Path(tmp), "zread")

            result = runtime.call_tool("read_file", {"path": "README.md"})
            runtime.close()

            self.assertEqual(result.content[0].text, "called read_file")
            self.assertIs(runtime.exit_stack.enter_task, runtime.exit_stack.close_task)


def _config(risk: ToolRiskLevel) -> McpConfig:
    return McpConfig(
        servers={"zread": McpServerConfig(name="zread", command="zread-mcp")},
        tools={
            "read_github_file": McpToolBinding(
                model_name="read_github_file",
                server="zread",
                mcp_tool_name="read_file",
                risk=risk,
                read_only=True,
            ),
            "write_github_file": McpToolBinding(
                model_name="write_github_file",
                server="zread",
                mcp_tool_name="write_file",
                risk=risk,
                read_only=False,
            )
        },
    )


class SameTaskRuntime(_ServerRuntime):
    def __init__(self, config: McpConfig, workspace: Path, server_name: str) -> None:
        self.exit_stack = SameTaskExitStack()
        super().__init__(config, workspace, server_name)

    async def _connect_server(self) -> _ConnectedServer:
        self.exit_stack.enter_task = asyncio.current_task()
        return _ConnectedServer(
            name=self.server_name,
            session=FakeAsyncMcpSession(),
            exit_stack=self.exit_stack,  # type: ignore[arg-type]
            tools=[
                SimpleNamespace(
                    name="read_file",
                    description="Read a file from GitHub.",
                    inputSchema={"type": "object"},
                )
            ],
        )


class SameTaskExitStack:
    def __init__(self) -> None:
        self.enter_task: asyncio.Task[Any] | None = None
        self.close_task: asyncio.Task[Any] | None = None

    async def aclose(self) -> None:
        self.close_task = asyncio.current_task()
        if self.enter_task is not self.close_task:
            raise RuntimeError("closed from a different task")


class FakeAsyncMcpSession:
    async def list_tools(self) -> Any:
        return SimpleNamespace(
            tools=[
                SimpleNamespace(
                    name="read_file",
                    description="Read a file from GitHub.",
                    inputSchema={"type": "object"},
                )
            ]
        )

    async def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        return SimpleNamespace(
            isError=False,
            content=[SimpleNamespace(type="text", text=f"called {tool_name}")],
        )


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

from mikucli.langgraph_runtime import GraphAgentSession
from mikucli.llm import AssistantMessage, TokenUsage, ToolCall
from mikucli.react import AgentSession
from mikucli.tools import ToolRegistry
from mikucli.workspace import Workspace


class FakeClient:
    def __init__(self, responses: list[AssistantMessage]) -> None:
        self.responses = responses
        self.requests: list[list[dict[str, Any]]] = []

    def chat(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        stream: bool = False,
    ) -> AssistantMessage:
        del model, tools, stream
        self.requests.append(messages)
        return self.responses.pop(0)


class FakeConsole:
    def __init__(self) -> None:
        self.answers: list[str] = []
        self.tool_results: list[tuple[str, bool, str]] = []

    def progress(self, message: str) -> None:
        del message

    def tool_request(self, name: str, arguments: dict[str, Any]) -> None:
        del name, arguments

    def tool_result(self, name: str, ok: bool, content: str, diff: str = "") -> None:
        del diff
        self.tool_results.append((name, ok, content))

    def answer(self, content: str) -> None:
        self.answers.append(content)

    def token_usage(self, usage: TokenUsage) -> None:
        del usage


def _responses() -> list[AssistantMessage]:
    return [
        AssistantMessage(
            content="",
            tool_calls=[ToolCall(id="call_1", name="list_files", arguments={})],
            raw={},
            token_usage=TokenUsage(total_tokens=10),
        ),
        AssistantMessage(
            content="done",
            tool_calls=[],
            raw={},
            token_usage=TokenUsage(total_tokens=20),
        ),
    ]


class GraphAgentSessionTests(unittest.TestCase):
    def test_langgraph_runtime_matches_native_tool_loop(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "README.md").write_text("hello", encoding="utf-8")
            native_client = FakeClient(_responses())
            graph_client = FakeClient(_responses())
            native_console = FakeConsole()
            graph_console = FakeConsole()
            tools = ToolRegistry(Workspace(root))
            native = AgentSession(
                client=native_client,  # type: ignore[arg-type]
                model="test-model",
                workspace=root,
                tools=tools,
                console=native_console,
            )
            graph = GraphAgentSession(
                client=graph_client,
                model="test-model",
                workspace=root,
                tools=tools,
                console=graph_console,
            )

            native_result = native.run_turn("list files")
            graph_result = graph.run_turn("list files")

            self.assertEqual(graph_result.final_answer, native_result.final_answer)
            self.assertEqual(graph_console.answers, native_console.answers)
            self.assertEqual(graph_console.tool_results, native_console.tool_results)
            self.assertEqual(len(graph_client.requests), 2)
            self.assertTrue(any(message["role"] == "tool" for message in graph_client.requests[1]))
            graph_log = json.loads(graph_result.log_path.read_text(encoding="utf-8"))
            self.assertEqual(graph_log["metadata"]["runtime"], "langgraph")

    def test_stop_request_prevents_model_call(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            client = FakeClient([])
            session = GraphAgentSession(
                client=client,
                model="test-model",
                workspace=root,
                tools=ToolRegistry(Workspace(root)),
                console=FakeConsole(),
            )

            result = session.run_turn("long task", stop_requested=lambda: True)

            self.assertEqual(result.final_answer, "Stopped by user.")
            self.assertEqual(client.requests, [])

    def test_json_fallback_tool_and_final_actions_follow_the_graph_loop(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            client = FakeClient(
                [
                    AssistantMessage(
                        content='{"tool":"list_files","arguments":{}}',
                        tool_calls=[],
                        raw={},
                        token_usage=TokenUsage(total_tokens=10),
                    ),
                    AssistantMessage(
                        content='{"final":"done"}',
                        tool_calls=[],
                        raw={},
                        token_usage=TokenUsage(total_tokens=20),
                    ),
                ]
            )
            session = GraphAgentSession(
                client=client,
                model="test-model",
                workspace=root,
                tools=ToolRegistry(Workspace(root)),
                console=FakeConsole(),
            )

            result = session.run_turn("list files")

            self.assertEqual(result.final_answer, "done")
            self.assertEqual(len(client.requests), 2)
            self.assertTrue(
                any(
                    message["role"] == "user"
                    and str(message["content"]).startswith("Tool result for list_files:")
                    for message in client.requests[1]
                )
            )

    def test_max_steps_stops_after_last_tool_execution(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            client = FakeClient(_responses()[:1])
            console = FakeConsole()
            session = GraphAgentSession(
                client=client,
                model="test-model",
                workspace=root,
                tools=ToolRegistry(Workspace(root)),
                console=console,
                max_steps=1,
            )

            result = session.run_turn("list files")

            self.assertEqual(
                result.final_answer,
                "Stopped because the session reached the maximum tool loop depth.",
            )
            self.assertEqual(len(client.requests), 1)
            self.assertEqual(console.answers, [result.final_answer])


if __name__ == "__main__":
    unittest.main()

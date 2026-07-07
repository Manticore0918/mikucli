from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

from mikucli.bench.runner import BenchmarkRunner
from mikucli.bench.tasks import all_benchmark_cases
from mikucli.bench.models import SessionMode
from mikucli.llm import AssistantMessage, TokenUsage, ToolCall


class FakeClient:
    def __init__(self, responses: list[AssistantMessage]) -> None:
        self.responses = responses
        self.requests: list[dict[str, Any]] = []

    def chat(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        stream: bool = False,
    ) -> AssistantMessage:
        self.requests.append({"messages": messages, "tools": tools})
        return self.responses.pop(0)


class BenchmarkCatalogTests(unittest.TestCase):
    def test_catalog_expands_to_twelve_cases(self) -> None:
        cases = all_benchmark_cases()

        self.assertEqual(len(cases), 12)
        self.assertIn("repo_inspection:built_in_single_agent", {case.id for case in cases})
        self.assertIn("repo_inspection:built_in_multi_agent", {case.id for case in cases})
        self.assertIn("mcp_tool_use:mcp_single_agent", {case.id for case in cases})
        self.assertIn("mcp_tool_use:mcp_multi_agent", {case.id for case in cases})
        self.assertEqual(
            {
                case.session_mode
                for case in cases
                if case.task.id == "mcp_tool_use"
            },
            {SessionMode.MCP_SINGLE_AGENT, SessionMode.MCP_MULTI_AGENT},
        )


class BenchmarkRunnerTests(unittest.TestCase):
    def test_single_agent_file_edit_case_records_results(self) -> None:
        case = _case("file_edit:built_in_single_agent")
        edited_readme = (
            "# Invoice Demo\n\n"
            "A small command-line invoice demo.\n\n"
            "## Usage\n\n"
            "Run `python app.py --demo`.\n\n"
            "Demo mode prints a sample invoice.\n"
        )
        client = FakeClient(
            [
                AssistantMessage(
                    content="",
                    tool_calls=[ToolCall(id="call_1", name="write_file", arguments={"path": "README.md", "content": edited_readme})],
                    raw={},
                    token_usage=TokenUsage(total_tokens=10),
                ),
                AssistantMessage(
                    content="Updated README.md only.",
                    tool_calls=[],
                    raw={},
                    token_usage=TokenUsage(total_tokens=10),
                ),
            ]
        )

        with tempfile.TemporaryDirectory() as tmp:
            results, result_path = BenchmarkRunner(root=Path(tmp), client=client, model="fake-model").run([case])

            self.assertTrue(result_path.exists())
            self.assertEqual(len(results), 1)
            result = results[0]
            self.assertTrue(result.passed)
            self.assertEqual(result.changed_paths, ["README.md"])
            self.assertEqual([call.name for call in result.tool_calls], ["write_file"])
            self.assertEqual(len(result.approvals), 1)
            self.assertTrue(result.approvals[0].approved)
            payload = json.loads(result_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["results"][0]["case_id"], "file_edit:built_in_single_agent")

    def test_code_search_case_requires_search_codebase(self) -> None:
        case = _case("code_search:built_in_single_agent")
        client = FakeClient(
            [
                AssistantMessage(
                    content="",
                    tool_calls=[
                        ToolCall(
                            id="call_1",
                            name="search_codebase",
                            arguments={"query": "priority invoice routing sentinel", "limit": 3},
                        )
                    ],
                    raw={},
                    token_usage=TokenUsage(total_tokens=10),
                ),
                AssistantMessage(
                    content="The sentinel is ORCHID-917 and priority invoices route to the amber queue.",
                    tool_calls=[],
                    raw={},
                    token_usage=TokenUsage(total_tokens=10),
                ),
            ]
        )

        with tempfile.TemporaryDirectory() as tmp:
            results, _ = BenchmarkRunner(root=Path(tmp), client=client, model="fake-model").run([case])

            self.assertTrue(results[0].passed)
            self.assertEqual([call.name for call in results[0].tool_calls], ["search_codebase"])


def _case(case_id: str):
    for case in all_benchmark_cases():
        if case.id == case_id:
            return case
    raise AssertionError(f"unknown case: {case_id}")


if __name__ == "__main__":
    unittest.main()

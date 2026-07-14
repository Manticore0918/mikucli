from __future__ import annotations

import json
import tempfile
import time
import unittest
from pathlib import Path
from typing import Any

from mikucli.evaluation.bench.runner import BenchmarkRunner
from mikucli.evaluation.bench.tasks import all_benchmark_cases
from mikucli.evaluation.bench.models import EvalPrice, SessionMode
from mikucli.llm import AssistantMessage, TokenUsage, ToolCall


class FakeClient:
    def __init__(self, responses: list[AssistantMessage], delay_seconds: float = 0.0) -> None:
        self.responses = responses
        self.delay_seconds = delay_seconds
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
        if self.delay_seconds:
            time.sleep(self.delay_seconds)
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
    def test_hallucination_check_resolves_bare_filenames_in_nested_directory(self) -> None:
        case = _case("repo_inspection:built_in_single_agent")
        client = FakeClient(
            [
                AssistantMessage(
                    content=(
                        "Acorn Ledger uses the `src/acorn/` package, which contains "
                        "`__init__.py` and `calculator.py`. Its test is `tests/test_calculator.py`."
                    ),
                    tool_calls=[],
                    raw={},
                    token_usage=TokenUsage(total_tokens=5),
                ),
            ]
        )

        with tempfile.TemporaryDirectory() as tmp:
            results, _, _ = BenchmarkRunner(root=Path(tmp), client=client, model="fake-model").run([case])

        path_check = next(
            check for check in results[0].hallucination_results if check.name == "answer_references_existing_files"
        )
        self.assertTrue(path_check.passed)
        self.assertEqual(path_check.messages, [])
        self.assertEqual(
            path_check.evidence["referenced_paths"],
            ["__init__.py", "calculator.py", "src/acorn", "tests/test_calculator.py"],
        )

    def test_hallucination_check_still_rejects_missing_bare_filename(self) -> None:
        case = _case("repo_inspection:built_in_single_agent")
        client = FakeClient(
            [
                AssistantMessage(
                    content=(
                        "Acorn Ledger uses `src/acorn/` and is tested by `tests/test_calculator.py`. "
                        "It also contains `missing.py`."
                    ),
                    tool_calls=[],
                    raw={},
                    token_usage=TokenUsage(total_tokens=5),
                ),
            ]
        )

        with tempfile.TemporaryDirectory() as tmp:
            results, _, _ = BenchmarkRunner(root=Path(tmp), client=client, model="fake-model").run([case])

        path_check = next(
            check for check in results[0].hallucination_results if check.name == "answer_references_existing_files"
        )
        self.assertFalse(path_check.passed)
        self.assertEqual(
            path_check.messages,
            ["final answer referenced missing workspace path: missing.py"],
        )

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
            results, result_path, report_path = BenchmarkRunner(root=Path(tmp), client=client, model="fake-model").run([case])

            self.assertTrue(result_path.exists())
            self.assertTrue(report_path.exists())
            self.assertEqual(len(results), 1)
            result = results[0]
            self.assertTrue(result.passed)
            self.assertEqual(result.changed_paths, ["README.md"])
            self.assertEqual([call.name for call in result.tool_calls], ["write_file"])
            self.assertEqual(len(result.approvals), 1)
            self.assertTrue(result.approvals[0].approved)
            self.assertEqual(result.metrics.tool_call_count, 1)
            self.assertEqual(result.metrics.cost.total_tokens, 20)
            self.assertEqual(result.metrics.model_retries, 0)
            self.assertGreaterEqual(result.metrics.elapsed_seconds, result.metrics.llm_latency_seconds)
            self.assertEqual(result.failure_reasons, [])
            payload = json.loads(result_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["summary"]["total_cases"], 1)
            self.assertEqual(payload["summary"]["success_rate"], 1.0)
            self.assertIn("agent_latency_seconds", payload["summary"])
            self.assertIn("llm_latency_seconds", payload["summary"])
            self.assertIn("agent_latency_seconds", payload["results"][0]["metrics"])
            self.assertIn("llm_latency_seconds", payload["results"][0]["metrics"])
            self.assertEqual(payload["results"][0]["case_id"], "file_edit:built_in_single_agent")
            report = report_path.read_text(encoding="utf-8")
            self.assertIn("Success rate: 100.0% (1/1)", report)
            self.assertIn("Total latency", report)
            self.assertIn("Agent latency", report)
            self.assertIn("LLM latency", report)

    def test_llm_latency_is_recorded_separately(self) -> None:
        case = _case("repo_inspection:built_in_single_agent")
        client = FakeClient(
            [
                AssistantMessage(
                    content="Acorn Ledger uses src/acorn and tests/test_calculator.py.",
                    tool_calls=[],
                    raw={},
                    token_usage=TokenUsage(total_tokens=5),
                ),
            ],
            delay_seconds=0.01,
        )

        with tempfile.TemporaryDirectory() as tmp:
            results, result_path, report_path = BenchmarkRunner(root=Path(tmp), client=client, model="fake-model").run([case])

            metrics = results[0].metrics
            self.assertGreater(metrics.llm_latency_seconds, 0)
            self.assertGreaterEqual(metrics.elapsed_seconds, metrics.llm_latency_seconds)
            payload = json.loads(result_path.read_text(encoding="utf-8"))
            self.assertGreater(payload["summary"]["llm_latency_seconds"], 0)
            self.assertIn("LLM latency", report_path.read_text(encoding="utf-8"))

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
            results, _, _ = BenchmarkRunner(root=Path(tmp), client=client, model="fake-model").run([case])

            self.assertTrue(results[0].passed)
            self.assertEqual([call.name for call in results[0].tool_calls], ["search_codebase"])

    def test_bug_fix_accepts_equivalent_percentage_tax_formula(self) -> None:
        case = _case("bug_fix:built_in_single_agent")
        client = FakeClient(
            [
                AssistantMessage(
                    content="",
                    tool_calls=[
                        ToolCall(
                            id="call_1",
                            name="write_file",
                            arguments={
                                "path": "src/shop/tax.py",
                                "content": "def add_tax(subtotal, tax_rate):\n    return subtotal * (1 + tax_rate)\n",
                            },
                        )
                    ],
                    raw={},
                    token_usage=TokenUsage(total_tokens=10),
                ),
                AssistantMessage(
                    content="Fixed add_tax and verified the tests.",
                    tool_calls=[],
                    raw={},
                    token_usage=TokenUsage(total_tokens=10),
                ),
            ]
        )

        with tempfile.TemporaryDirectory() as tmp:
            results, _, _ = BenchmarkRunner(root=Path(tmp), client=client, model="fake-model").run([case])

            self.assertTrue(results[0].passed)

    def test_price_generates_estimated_spend(self) -> None:
        case = _case("repo_inspection:built_in_single_agent")
        client = FakeClient(
            [
                AssistantMessage(
                    content="Acorn Ledger uses src/acorn and tests/test_calculator.py.",
                    tool_calls=[],
                    raw={},
                    token_usage=TokenUsage(prompt_tokens=1000, completion_tokens=250, total_tokens=1250),
                ),
            ]
        )

        with tempfile.TemporaryDirectory() as tmp:
            results, result_path, _ = BenchmarkRunner(
                root=Path(tmp),
                client=client,
                model="fake-model",
                price=EvalPrice(prompt_token_price_per_million=2.0, completion_token_price_per_million=4.0),
            ).run([case])

            result = results[0]
            self.assertEqual(result.metrics.cost.prompt_tokens, 1000)
            self.assertEqual(result.metrics.cost.completion_tokens, 250)
            self.assertIsNotNone(result.metrics.estimated_spend)
            self.assertEqual(result.metrics.estimated_spend.total, 0.003)
            payload = json.loads(result_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["summary"]["estimated_spend"]["total"], 0.003)

    def test_failure_reasons_are_structured(self) -> None:
        case = _case("repo_inspection:built_in_single_agent")
        client = FakeClient(
            [
                AssistantMessage(
                    content="Wrong answer.",
                    tool_calls=[],
                    raw={},
                    token_usage=TokenUsage(total_tokens=5),
                ),
            ]
        )

        with tempfile.TemporaryDirectory() as tmp:
            results, result_path, report_path = BenchmarkRunner(root=Path(tmp), client=client, model="fake-model").run([case])

            result = results[0]
            self.assertFalse(result.passed)
            self.assertTrue(result.failure_reasons)
            self.assertEqual(result.failure_reasons[0].category, "check_failed")
            payload = json.loads(result_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["results"][0]["failure_reasons"][0]["category"], "check_failed")
            self.assertIn("## Failure Reasons", report_path.read_text(encoding="utf-8"))

    def test_stop_requested_writes_partial_report(self) -> None:
        cases = [
            _case("repo_inspection:built_in_single_agent"),
            _case("file_edit:built_in_single_agent"),
        ]
        client = FakeClient(
            [
                AssistantMessage(
                    content="Acorn Ledger uses src/acorn and tests/test_calculator.py.",
                    tool_calls=[],
                    raw={},
                    token_usage=TokenUsage(total_tokens=5),
                ),
            ]
        )
        checks = 0

        def stop_requested() -> bool:
            nonlocal checks
            checks += 1
            return checks > 1

        with tempfile.TemporaryDirectory() as tmp:
            results, result_path, report_path = BenchmarkRunner(
                root=Path(tmp),
                client=client,
                model="fake-model",
                stop_requested=stop_requested,
            ).run(cases)

            self.assertEqual([result.case_id for result in results], ["repo_inspection:built_in_single_agent"])
            payload = json.loads(result_path.read_text(encoding="utf-8"))
            self.assertTrue(payload["summary"]["stopped"])
            self.assertIn("Stopped: yes", report_path.read_text(encoding="utf-8"))

    def test_on_case_finished_callback_receives_results(self) -> None:
        case = _case("repo_inspection:built_in_single_agent")
        client = FakeClient(
            [
                AssistantMessage(
                    content="Acorn Ledger uses src/acorn and tests/test_calculator.py.",
                    tool_calls=[],
                    raw={},
                    token_usage=TokenUsage(total_tokens=5),
                ),
            ]
        )
        finished: list[str] = []

        with tempfile.TemporaryDirectory() as tmp:
            BenchmarkRunner(
                root=Path(tmp),
                client=client,
                model="fake-model",
                on_case_finished=lambda result: finished.append(result.case_id),
            ).run([case])

            self.assertEqual(finished, ["repo_inspection:built_in_single_agent"])

    def test_on_case_started_callback_receives_cases(self) -> None:
        case = _case("repo_inspection:built_in_single_agent")
        client = FakeClient(
            [
                AssistantMessage(
                    content="Acorn Ledger uses src/acorn and tests/test_calculator.py.",
                    tool_calls=[],
                    raw={},
                    token_usage=TokenUsage(total_tokens=5),
                ),
            ]
        )
        started: list[str] = []

        with tempfile.TemporaryDirectory() as tmp:
            BenchmarkRunner(
                root=Path(tmp),
                client=client,
                model="fake-model",
                on_case_started=lambda case: started.append(case.id),
            ).run([case])

            self.assertEqual(started, ["repo_inspection:built_in_single_agent"])


def _case(case_id: str):
    for case in all_benchmark_cases():
        if case.id == case_id:
            return case
    raise AssertionError(f"unknown case: {case_id}")


if __name__ == "__main__":
    unittest.main()

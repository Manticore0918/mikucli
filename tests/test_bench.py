from __future__ import annotations

import json
import os
import sqlite3
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from typing import Any
from unittest.mock import patch

from mikucli.evaluation.bench.__main__ import _sum_optional
from mikucli.evaluation.bench.metrics import cost_from_usage, sum_costs
from mikucli.evaluation.bench.models import EvalCost, EvalPrice, SessionMode
from mikucli.evaluation.bench.runner import BenchmarkRunner
from mikucli.evaluation.bench.sandbox import isolated_function_outputs
from mikucli.evaluation.bench.support import approval_recorder
from mikucli.evaluation.bench.tasks import all_benchmark_cases
from mikucli.llm import AssistantMessage, TokenUsage, ToolCall
from mikucli.tools import ToolApprovalRequest, ToolRiskLevel


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


class BenchmarkSecurityTests(unittest.TestCase):
    def test_approval_recorder_rejects_arbitrary_shell_commands(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            approvals = []
            confirm = approval_recorder(workspace, approvals)

            external_read = confirm(
                ToolApprovalRequest(
                    tool_name="run_shell",
                    risk_level=ToolRiskLevel.HIGH,
                    workspace=str(workspace),
                    summary="Read an external file",
                    details="reason: inspect host\ncommand: Get-Content C:\\Windows\\win.ini",
                )
            )
            test_run = confirm(
                ToolApprovalRequest(
                    tool_name="run_shell",
                    risk_level=ToolRiskLevel.HIGH,
                    workspace=str(workspace),
                    summary="Run tests",
                    details="reason: verify fix\ncommand: python -m unittest discover -s tests",
                )
            )

        self.assertFalse(external_read)
        self.assertTrue(test_run)
        self.assertEqual([record.approved for record in approvals], [False, True])

    def test_function_check_isolated_from_system_exit_and_parent_environment(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            target = workspace / "target.py"
            target.write_text(
                "import os\n"
                "def exits():\n    raise SystemExit('stop')\n"
                "def secret():\n    return os.environ.get('MIKUCLI_BENCH_SECRET')\n",
                encoding="utf-8",
            )

            exit_messages = isolated_function_outputs(workspace, target, "exits", (((), None),))
            with patch.dict(os.environ, {"MIKUCLI_BENCH_SECRET": "must-not-leak"}):
                secret_messages = isolated_function_outputs(workspace, target, "secret", (((), None),))

        self.assertEqual(exit_messages, ["exits() raised SystemExit: stop"])
        self.assertEqual(secret_messages, [])

    def test_function_check_cannot_read_outside_workspace(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            workspace.mkdir()
            secret = root / "secret.txt"
            secret.write_text("host secret", encoding="utf-8")
            target = workspace / "target.py"
            target.write_text(
                f"def read_external():\n    return open({str(secret)!r}, encoding='utf-8').read()\n",
                encoding="utf-8",
            )

            messages = isolated_function_outputs(workspace, target, "read_external", (((), "host secret"),))

        self.assertEqual(len(messages), 1)
        self.assertIn("PermissionError", messages[0])


class BenchmarkMetricTests(unittest.TestCase):
    def test_partial_usage_is_reported_as_unknown(self) -> None:
        cost = cost_from_usage(
            [
                TokenUsage(prompt_tokens=100, total_tokens=100),
                TokenUsage(prompt_tokens=None, total_tokens=50),
            ]
        )
        combined = sum_costs([EvalCost(total_tokens=100), EvalCost(total_tokens=None)])

        self.assertIsNone(cost.prompt_tokens)
        self.assertEqual(cost.total_tokens, 150)
        self.assertIsNone(combined.total_tokens)
        self.assertIsNone(_sum_optional([1, None]))


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
        self.assertFalse(results[0].passed)
        self.assertFalse(path_check.passed)
        self.assertEqual(
            path_check.messages,
            ["final answer referenced missing workspace path: missing.py"],
        )

    def test_tool_claim_check_ignores_negated_mentions(self) -> None:
        case = _case("repo_inspection:built_in_single_agent")
        client = FakeClient(
            [
                AssistantMessage(
                    content=(
                        "Acorn Ledger lives at src/acorn and its tests are at tests/test_calculator.py. "
                        "I inspected it without using write_file."
                    ),
                    tool_calls=[],
                    raw={},
                    token_usage=TokenUsage(total_tokens=5),
                ),
            ]
        )

        with tempfile.TemporaryDirectory() as tmp:
            results, _, _ = BenchmarkRunner(root=Path(tmp), client=client, model="fake-model").run([case])

        tool_check = next(
            check for check in results[0].hallucination_results if check.name == "tool_claim_has_trace"
        )
        self.assertTrue(tool_check.passed)
        self.assertEqual(tool_check.messages, [])
        self.assertTrue(results[0].passed)

    def test_tool_claim_check_ignores_hypothetical_mentions(self) -> None:
        case = _case("repo_inspection:built_in_single_agent")
        client = FakeClient(
            [
                AssistantMessage(
                    content=(
                        "Acorn Ledger lives at src/acorn and its tests are at tests/test_calculator.py. "
                        "I considered run_shell but chose read-only tools instead."
                    ),
                    tool_calls=[],
                    raw={},
                    token_usage=TokenUsage(total_tokens=5),
                ),
            ]
        )

        with tempfile.TemporaryDirectory() as tmp:
            results, _, _ = BenchmarkRunner(root=Path(tmp), client=client, model="fake-model").run([case])

        tool_check = next(
            check for check in results[0].hallucination_results if check.name == "tool_claim_has_trace"
        )
        self.assertTrue(tool_check.passed)
        self.assertEqual(tool_check.messages, [])
        self.assertTrue(results[0].passed)

    def test_tool_claim_check_still_flags_positive_untraced_claims(self) -> None:
        case = _case("repo_inspection:built_in_single_agent")
        client = FakeClient(
            [
                AssistantMessage(
                    content=(
                        "Acorn Ledger lives at src/acorn and its tests are at tests/test_calculator.py. "
                        "I used search_codebase to find the priority sentinel."
                    ),
                    tool_calls=[],
                    raw={},
                    token_usage=TokenUsage(total_tokens=5),
                ),
            ]
        )

        with tempfile.TemporaryDirectory() as tmp:
            results, _, _ = BenchmarkRunner(root=Path(tmp), client=client, model="fake-model").run([case])

        tool_check = next(
            check for check in results[0].hallucination_results if check.name == "tool_claim_has_trace"
        )
        self.assertFalse(tool_check.passed)
        self.assertEqual(tool_check.messages, ["final answer claimed use of tool 'search_codebase', but that tool was not recorded."])
        self.assertFalse(results[0].passed)

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
            self.assertEqual(result.failure_reasons, [])
            payload = json.loads(result_path.read_text(encoding="utf-8"))
            self.assertLessEqual(
                datetime.fromisoformat(payload["started_at"]),
                datetime.fromisoformat(payload["ended_at"]),
            )
            self.assertEqual(payload["summary"]["total_cases"], 1)
            self.assertEqual(payload["summary"]["success_rate"], 1.0)
            self.assertIn("elapsed_seconds", payload["summary"])
            self.assertIn("elapsed_seconds", payload["results"][0]["metrics"])
            self.assertNotIn("agent_latency_seconds", payload["summary"])
            self.assertNotIn("llm_latency_seconds", payload["summary"])
            self.assertNotIn("agent_latency_seconds", payload["results"][0]["metrics"])
            self.assertNotIn("llm_latency_seconds", payload["results"][0]["metrics"])
            self.assertEqual(payload["results"][0]["case_id"], "file_edit:built_in_single_agent")
            report = report_path.read_text(encoding="utf-8")
            self.assertIn("Success rate: 100.0% (1/1)", report)
            self.assertIn("End-to-end latency", report)
            self.assertNotIn("Agent latency", report)
            self.assertNotIn("LLM latency", report)

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

    def test_bug_fix_can_run_approved_tests_through_isolated_shell_wrapper(self) -> None:
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
                    content="",
                    tool_calls=[
                        ToolCall(
                            id="call_2",
                            name="run_shell",
                            arguments={
                                "command": "python -m unittest discover -s tests",
                                "reason": "verify the fix",
                            },
                        )
                    ],
                    raw={},
                    token_usage=TokenUsage(total_tokens=10),
                ),
                AssistantMessage(
                    content="Fixed add_tax and the tests passed.",
                    tool_calls=[],
                    raw={},
                    token_usage=TokenUsage(total_tokens=10),
                ),
            ]
        )

        with tempfile.TemporaryDirectory() as tmp:
            results, _, _ = BenchmarkRunner(root=Path(tmp), client=client, model="fake-model").run([case])

        self.assertTrue(results[0].passed)
        self.assertEqual([call.name for call in results[0].tool_calls], ["write_file", "run_shell"])
        self.assertTrue(all(approval.approved for approval in results[0].approvals))

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

    def test_failed_tool_correctness_signal_fails_the_case(self) -> None:
        case = _case("repo_inspection:built_in_single_agent")
        client = FakeClient(
            [
                AssistantMessage(
                    content="",
                    tool_calls=[
                        ToolCall(
                            id="call_1",
                            name="run_shell",
                            arguments={"command": "sudo cat /etc/passwd", "reason": "inspect host"},
                        )
                    ],
                    raw={},
                    token_usage=TokenUsage(total_tokens=5),
                ),
                AssistantMessage(
                    content="Acorn Ledger uses src/acorn and tests/test_calculator.py.",
                    tool_calls=[],
                    raw={},
                    token_usage=TokenUsage(total_tokens=5),
                ),
            ]
        )

        with tempfile.TemporaryDirectory() as tmp:
            results, _, _ = BenchmarkRunner(root=Path(tmp), client=client, model="fake-model").run([case])

        failed = [check.name for check in results[0].tool_correctness_results if not check.passed]
        self.assertFalse(results[0].passed)
        self.assertIn("high_risk_tools_approved", failed)

    def test_observability_import_failure_does_not_discard_results(self) -> None:
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

        with tempfile.TemporaryDirectory() as tmp:
            runner = BenchmarkRunner(root=Path(tmp), client=client, model="fake-model")
            with patch.object(
                runner.trace_store,
                "import_eval_report",
                side_effect=sqlite3.OperationalError("database unavailable"),
            ), self.assertWarnsRegex(RuntimeWarning, "observability import failed"):
                results, result_path, report_path = runner.run([case])

            self.assertTrue(results[0].passed)
            self.assertTrue(result_path.exists())
            self.assertTrue(report_path.exists())

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
            self.assertEqual(results[0].final_answer, "Stopped by user.")
            self.assertEqual(client.requests, [])
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

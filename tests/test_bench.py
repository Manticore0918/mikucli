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
from mikucli.evaluation.bench.catalog import expand_benchmark_cases, validate_benchmark_tasks
from mikucli.evaluation.bench.metrics import cost_from_usage, sum_costs
from mikucli.evaluation.bench.metadata import build_run_metadata
from mikucli.evaluation.bench.models import (
    BenchmarkCase,
    BenchmarkTask,
    EvalCost,
    EvalPrice,
    SessionMode,
    TaskSetup,
)
from mikucli.evaluation.bench.runner import BenchmarkRunner
from mikucli.evaluation.bench.sandbox import (
    _MAX_CAPTURE_BYTES,
    _run_isolated_python,
    isolated_function_outputs,
    run_isolated_unittest,
)
from mikucli.evaluation.bench.support import approval_recorder
from mikucli.evaluation.bench.tasks import all_benchmark_cases
from mikucli.llm import AssistantMessage, TokenUsage, ToolCall
from mikucli.react import SessionResult
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

    def test_catalog_rejects_duplicate_task_ids(self) -> None:
        task = BenchmarkTask(
            id="duplicate",
            title="Duplicate",
            prompt="Do the task.",
            setup=lambda workspace: TaskSetup(),
            checks=(lambda context: [],),
            session_modes=(SessionMode.BUILT_IN_SINGLE_AGENT,),
        )

        with self.assertRaisesRegex(Exception, "duplicate benchmark task id"):
            validate_benchmark_tasks([task, task])

    def test_catalog_rejects_duplicate_session_modes(self) -> None:
        task = BenchmarkTask(
            id="duplicate_mode",
            title="Duplicate mode",
            prompt="Do the task.",
            setup=lambda workspace: TaskSetup(),
            checks=(lambda context: [],),
            session_modes=(
                SessionMode.BUILT_IN_SINGLE_AGENT,
                SessionMode.BUILT_IN_SINGLE_AGENT,
            ),
        )

        with self.assertRaisesRegex(Exception, "duplicate session modes"):
            expand_benchmark_cases([task])


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

    def test_function_check_preserves_tuple_arguments(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            target = workspace / "target.py"
            target.write_text(
                "def accepts_tuple(value):\n    return isinstance(value, tuple)\n",
                encoding="utf-8",
            )

            messages = isolated_function_outputs(
                workspace,
                target,
                "accepts_tuple",
                (((((1, 2),), True)),),
            )

        self.assertEqual(messages, [])

    def test_function_check_cannot_replace_parent_comparison_or_spoof_equality(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            target = workspace / "target.py"
            target.write_text(
                "import __main__\n"
                "__main__.values_equal = lambda actual, expected: True\n"
                "class AlwaysEqual:\n"
                "    def __eq__(self, other):\n"
                "        return True\n"
                "def wrong_number():\n"
                "    return -999\n"
                "def spoofed_object():\n"
                "    return AlwaysEqual()\n",
                encoding="utf-8",
            )

            wrong_number = isolated_function_outputs(
                workspace, target, "wrong_number", (((), 42),)
            )
            spoofed_object = isolated_function_outputs(
                workspace, target, "spoofed_object", (((), 42),)
            )

        self.assertTrue(wrong_number)
        self.assertTrue(spoofed_object)

    def test_unittest_check_cannot_replace_the_runner_during_test_import(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            (workspace / "tests").mkdir()
            (workspace / "src").mkdir()
            (workspace / "tests" / "test_broken.py").write_text(
                "import unittest\n"
                "class FakeResult:\n"
                "    testsRun = 1\n"
                "    failures = []\n"
                "    errors = []\n"
                "    unexpectedSuccesses = []\n"
                "unittest.TextTestRunner.run = lambda self, suite: FakeResult()\n"
                "class BrokenTest(unittest.TestCase):\n"
                "    def test_broken(self):\n"
                "        self.assertEqual(1, 2)\n",
                encoding="utf-8",
            )

            completed = run_isolated_unittest(workspace)

        self.assertNotEqual(completed.returncode, 0)

    def test_isolated_process_capture_is_bounded(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = _run_isolated_python(
                Path(tmp),
                ["import sys; sys.stdout.write('x' * 200000)"],
                timeout_seconds=5,
            )

        self.assertEqual(result.returncode, 0)
        self.assertIn("output bytes omitted", result.stdout)
        self.assertLessEqual(len(result.stdout.encode("utf-8")), _MAX_CAPTURE_BYTES + 100)


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

    def test_provider_settings_change_configuration_fingerprint_without_leaking_query(self) -> None:
        class Client:
            def __init__(self, base_url: str) -> None:
                self.base_url = base_url

        case = _case("repo_inspection:built_in_single_agent")
        common = {
            "root": Path.cwd(),
            "model": "fake-model",
            "max_steps": 30,
            "context_window_tokens": 128000,
            "cases": [case],
            "repeat_count": 1,
            "shuffle_seed": None,
            "memory_retriever": None,
            "price": None,
        }
        first = build_run_metadata(client=Client("https://one.example/v1?token=secret"), **common)
        second = build_run_metadata(client=Client("https://two.example/v1"), **common)

        self.assertNotEqual(
            first["configuration_fingerprint"], second["configuration_fingerprint"]
        )
        self.assertEqual(
            first["configuration"]["client"]["base_url"], "https://one.example/v1"
        )

    def test_price_rejects_negative_and_non_finite_values(self) -> None:
        for value in (-1.0, float("inf"), float("nan")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                EvalPrice(prompt_token_price_per_million=value)


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

    def test_tool_claim_check_scopes_negation_to_its_clause(self) -> None:
        case = _case("repo_inspection:built_in_single_agent")
        client = FakeClient(
            [
                AssistantMessage(
                    content=(
                        "Acorn Ledger lives at src/acorn and its tests are at tests/test_calculator.py. "
                        "I did not use read_file, but I used run_shell."
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
        self.assertIn("run_shell", tool_check.messages[0])
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

    def test_file_edit_case_detects_extra_mikucli_file(self) -> None:
        case = _case("file_edit:built_in_single_agent")
        edited_readme = (
            "# Invoice Demo\n\nA small command-line invoice demo.\n\n"
            "## Usage\n\npython app.py --demo\n\n"
            "Demo mode prints a sample invoice.\n"
        )
        client = FakeClient(
            [
                AssistantMessage(
                    content="",
                    tool_calls=[
                        ToolCall("call_1", "write_file", {"path": "README.md", "content": edited_readme}),
                        ToolCall(
                            "call_2",
                            "write_file",
                            {"path": ".mikucli/extra.txt", "content": "must be tracked"},
                        ),
                    ],
                    raw={},
                    token_usage=TokenUsage(total_tokens=10),
                ),
                AssistantMessage(
                    content="Updated README.md.",
                    tool_calls=[],
                    raw={},
                    token_usage=TokenUsage(total_tokens=10),
                ),
            ]
        )

        with tempfile.TemporaryDirectory() as tmp:
            results, _, _ = BenchmarkRunner(root=Path(tmp), client=client, model="fake-model").run([case])

        self.assertFalse(results[0].passed)
        self.assertEqual(results[0].changed_paths, [".mikucli/extra.txt", "README.md"])

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

    def test_required_tool_must_succeed_and_failed_tool_must_be_recovered(self) -> None:
        case = _case("code_search:built_in_single_agent")
        client = FakeClient(
            [
                AssistantMessage(
                    content="",
                    tool_calls=[ToolCall("call_1", "search_codebase", {"query": "", "limit": 3})],
                    raw={},
                    token_usage=TokenUsage(total_tokens=10),
                ),
                AssistantMessage(
                    content="The sentinel is ORCHID-917 and the queue is amber queue.",
                    tool_calls=[],
                    raw={},
                    token_usage=TokenUsage(total_tokens=10),
                ),
            ]
        )

        with tempfile.TemporaryDirectory() as tmp:
            results, _, _ = BenchmarkRunner(
                root=Path(tmp), client=client, model="fake-model"
            ).run([case])

        self.assertFalse(results[0].passed)
        failed = {
            check.name
            for check in results[0].tool_correctness_results
            if not check.passed
        }
        self.assertIn("required_tools_called", failed)
        self.assertIn("failed_tools_recovered", failed)
        self.assertEqual(results[0].metrics.model_retries, 1)

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

    def test_bug_fix_rejects_forged_result_and_abrupt_exit(self) -> None:
        case = _case("bug_fix:built_in_single_agent")
        malicious = (
            "import os, sys\n"
            "if len(sys.argv) > 2:\n"
            "    print(sys.argv[2] + '{\\\"messages\\\": []}', flush=True)\n"
            "os._exit(0)\n"
        )
        client = FakeClient(
            [
                AssistantMessage(
                    content="",
                    tool_calls=[
                        ToolCall(
                            "call_1",
                            "write_file",
                            {"path": "src/shop/tax.py", "content": malicious},
                        )
                    ],
                    raw={},
                    token_usage=TokenUsage(total_tokens=10),
                ),
                AssistantMessage(
                    content="Fixed add_tax and tests passed.",
                    tool_calls=[],
                    raw={},
                    token_usage=TokenUsage(total_tokens=10),
                ),
            ]
        )

        with tempfile.TemporaryDirectory() as tmp:
            results, _, _ = BenchmarkRunner(root=Path(tmp), client=client, model="fake-model").run([case])

        self.assertFalse(results[0].passed)
        self.assertTrue(any(not check.passed for check in results[0].check_results))

    def test_test_repair_rejects_zero_unittest_tests_and_commented_expectation(self) -> None:
        case = _case("test_repair:built_in_single_agent")
        pytest_only = (
            "from shop.discounts import apply_discount\n\n"
            "def test_discount():\n"
            "    assert apply_discount(100, 0.25) == 25.0\n\n"
            "# self.assertEqual(apply_discount(100, 0.25), 75.0)\n"
        )
        client = FakeClient(
            [
                AssistantMessage(
                    content="",
                    tool_calls=[
                        ToolCall(
                            "call_1",
                            "write_file",
                            {"path": "tests/test_discounts.py", "content": pytest_only},
                        )
                    ],
                    raw={},
                    token_usage=TokenUsage(total_tokens=10),
                ),
                AssistantMessage(
                    content="Repaired the expectation and tests passed.",
                    tool_calls=[],
                    raw={},
                    token_usage=TokenUsage(total_tokens=10),
                ),
            ]
        )

        with tempfile.TemporaryDirectory() as tmp:
            results, _, _ = BenchmarkRunner(root=Path(tmp), client=client, model="fake-model").run([case])

        self.assertFalse(results[0].passed)
        failed = {check.name for check in results[0].check_results if not check.passed}
        self.assertIn("_tests_pass", failed)
        self.assertIn("python_contains_expression_tests_test_discounts.py", failed)

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

    def test_tool_cleanup_failure_is_reported_without_discarding_artifacts(self) -> None:
        class BrokenTools:
            def schemas(self):
                return []

            def read_only_tool_names(self):
                return set()

            def requires_approval(self, name, arguments=None):
                return False

            def invoke(self, name, arguments):
                raise AssertionError("unused")

            def close(self):
                raise RuntimeError("close failed")

        class Session:
            def run_turn(self, prompt, stop_requested=None):
                return SessionResult(final_answer="ok", log_path=Path("missing-log.json"))

        task = BenchmarkTask(
            id="cleanup_failure",
            title="Cleanup failure",
            prompt="noop",
            setup=lambda workspace: TaskSetup(),
            checks=(),
            session_modes=(SessionMode.BUILT_IN_SINGLE_AGENT,),
        )
        case = BenchmarkCase(
            id="cleanup_failure:built_in_single_agent",
            task=task,
            session_mode=SessionMode.BUILT_IN_SINGLE_AGENT,
        )

        with tempfile.TemporaryDirectory() as tmp:
            runner = BenchmarkRunner(root=Path(tmp), client=object(), model="fake-model")
            with patch.object(runner, "_build_tools", return_value=BrokenTools()), patch.object(
                runner,
                "_build_session",
                return_value=Session(),
            ):
                results, result_path, report_path = runner.run([case])

            self.assertFalse(results[0].passed)
            self.assertIn("cleanup_failed", {reason.category for reason in results[0].failure_reasons})
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

            self.assertEqual(results, [])
            self.assertEqual(client.requests, [])
            payload = json.loads(result_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["status"], "stopped")
            self.assertEqual(payload["summary"]["planned_trials"], 2)
            self.assertEqual(payload["summary"]["completed_trials"], 0)
            self.assertEqual(payload["summary"]["incomplete_trials"], 2)
            self.assertTrue(payload["summary"]["stopped"])
            self.assertIn("Stopped: yes", report_path.read_text(encoding="utf-8"))

    def test_repeat_runs_each_case_and_reports_distribution(self) -> None:
        case = _case("repo_inspection:built_in_single_agent")
        response = "Acorn Ledger uses src/acorn and tests/test_calculator.py."
        client = FakeClient(
            [
                AssistantMessage(
                    content=response,
                    tool_calls=[],
                    raw={},
                    token_usage=TokenUsage(total_tokens=5),
                ),
                AssistantMessage(
                    content=response,
                    tool_calls=[],
                    raw={},
                    token_usage=TokenUsage(total_tokens=5),
                ),
                AssistantMessage(
                    content=response,
                    tool_calls=[],
                    raw={},
                    token_usage=TokenUsage(total_tokens=5),
                ),
            ]
        )

        with tempfile.TemporaryDirectory() as tmp:
            results, result_path, _ = BenchmarkRunner(
                root=Path(tmp),
                client=client,
                model="fake-model",
                repeat_count=3,
                shuffle_seed=7,
            ).run([case])
            payload = json.loads(result_path.read_text(encoding="utf-8"))

        self.assertEqual(sorted(result.trial_index for result in results), [1, 2, 3])
        self.assertEqual({result.trial_count for result in results}, {3})
        self.assertEqual(payload["status"], "complete")
        self.assertEqual(payload["summary"]["planned_trials"], 3)
        self.assertEqual(payload["summary"]["completed_trials"], 3)
        self.assertIsNotNone(payload["summary"]["median_elapsed_seconds"])
        self.assertIsNotNone(payload["summary"]["p95_elapsed_seconds"])
        self.assertLess(payload["summary"]["pass_rate_ci95_low"], 1.0)

    def test_checkpoint_exists_before_case_and_resume_skips_completed_trials(self) -> None:
        case = _case("repo_inspection:built_in_single_agent")
        response = "Acorn Ledger uses src/acorn and tests/test_calculator.py."
        client = FakeClient(
            [
                AssistantMessage(
                    content=response,
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
            return checks > 3

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first_results, checkpoint_path, _ = BenchmarkRunner(
                root=root,
                client=client,
                model="fake-model",
                repeat_count=2,
                stop_requested=stop_requested,
            ).run([case])
            checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            resumed_client = FakeClient(
                [
                    AssistantMessage(
                        content=response,
                        tool_calls=[],
                        raw={},
                        token_usage=TokenUsage(total_tokens=5),
                    ),
                ]
            )
            resumed_results, resumed_path, _ = BenchmarkRunner(
                root=root,
                client=resumed_client,
                model="fake-model",
                repeat_count=2,
                resume_path=checkpoint_path,
            ).run([case])
            resumed_payload = json.loads(resumed_path.read_text(encoding="utf-8"))

        self.assertEqual(len(first_results), 1)
        self.assertEqual(checkpoint["status"], "stopped")
        self.assertEqual(len(resumed_client.requests), 1)
        self.assertEqual(sorted(result.trial_index for result in resumed_results), [1, 2])
        self.assertEqual(resumed_payload["status"], "complete")

    def test_resume_rejects_configuration_mismatch(self) -> None:
        case = _case("repo_inspection:built_in_single_agent")
        client = FakeClient([])

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runner = BenchmarkRunner(
                root=root,
                client=client,
                model="fake-model",
                repeat_count=1,
                stop_requested=lambda: True,
            )
            _, checkpoint_path, _ = runner.run([case])
            with self.assertRaisesRegex(Exception, "configuration does not match"):
                BenchmarkRunner(
                    root=root,
                    client=FakeClient([]),
                    model="different-model",
                    repeat_count=1,
                    resume_path=checkpoint_path,
                ).run([case])

    def test_resume_rejects_duplicate_trial_records(self) -> None:
        case = _case("repo_inspection:built_in_single_agent")
        response = "Acorn Ledger uses src/acorn and tests/test_calculator.py."
        checks = 0

        def stop_requested() -> bool:
            nonlocal checks
            checks += 1
            return checks > 3

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runner = BenchmarkRunner(
                root=root,
                client=FakeClient(
                    [
                        AssistantMessage(
                            content=response,
                            tool_calls=[],
                            raw={},
                            token_usage=TokenUsage(total_tokens=5),
                        )
                    ]
                ),
                model="fake-model",
                repeat_count=2,
                stop_requested=stop_requested,
            )
            _, checkpoint_path, _ = runner.run([case])
            payload = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            payload["results"].append(json.loads(json.dumps(payload["results"][0])))
            payload["summary"]["completed_trials"] = 2
            payload["summary"]["incomplete_trials"] = 0
            payload["summary"]["total_cases"] = 2
            payload["summary"]["passed_cases"] = 2
            payload["summary"]["success_rate"] = 1.0
            checkpoint_path.write_text(json.dumps(payload), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "duplicate benchmark result"):
                BenchmarkRunner(
                    root=root,
                    client=FakeClient([]),
                    model="fake-model",
                    repeat_count=2,
                    resume_path=checkpoint_path,
                ).run([case])

    def test_resume_rejects_inconsistent_summary_counts(self) -> None:
        case = _case("repo_inspection:built_in_single_agent")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runner = BenchmarkRunner(
                root=root,
                client=FakeClient([]),
                model="fake-model",
                stop_requested=lambda: True,
            )
            _, checkpoint_path, _ = runner.run([case])
            payload = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            payload["summary"]["completed_trials"] = 1
            checkpoint_path.write_text(json.dumps(payload), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "completed counts do not match"):
                BenchmarkRunner(
                    root=root,
                    client=FakeClient([]),
                    model="fake-model",
                    resume_path=checkpoint_path,
                ).run([case])

    def test_report_contains_reproducibility_metadata_and_sandbox_capabilities(self) -> None:
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
            _, result_path, _ = BenchmarkRunner(
                root=Path(tmp),
                client=client,
                model="fake-model",
            ).run([case])
            payload = json.loads(result_path.read_text(encoding="utf-8"))

        self.assertEqual(payload["schema_version"], 2)
        self.assertEqual(len(payload["suite"]["fingerprint"]), 64)
        self.assertEqual(len(payload["configuration_fingerprint"]), 64)
        self.assertFalse(payload["configuration"]["sandbox"]["os_enforced"])
        self.assertEqual(payload["configuration"]["client"]["type"], f"{FakeClient.__module__}.{FakeClient.__qualname__}")
        self.assertIn("python", payload["environment"])

    def test_pending_trial_workspace_is_recreated_before_setup(self) -> None:
        case = _case("file_edit:built_in_single_agent")
        with tempfile.TemporaryDirectory() as tmp:
            runner = BenchmarkRunner(root=Path(tmp), client=FakeClient([]), model="fake-model")
            workspace = runner.workspaces_root / "file_edit_built_in_single_agent"
            workspace.mkdir(parents=True)
            runner.stop_requested = lambda: True
            _, checkpoint_path, _ = runner.run([case])
            (workspace / "stale.txt").write_text("from crashed trial", encoding="utf-8")

            resumed = BenchmarkRunner(
                root=Path(tmp),
                client=FakeClient(
                    [
                        AssistantMessage(
                            content="done",
                            tool_calls=[],
                            raw={},
                            token_usage=TokenUsage(total_tokens=5),
                        )
                    ]
                ),
                model="fake-model",
                resume_path=checkpoint_path,
            )
            results, _, _ = resumed.run([case])

        self.assertNotIn("stale.txt", results[0].changed_paths)
        self.assertFalse((Path(results[0].workspace) / "stale.txt").exists())

    def test_require_os_sandbox_refuses_audit_hook_backend(self) -> None:
        case = _case("repo_inspection:built_in_single_agent")
        with tempfile.TemporaryDirectory() as tmp, self.assertRaisesRegex(
            Exception,
            "OS-enforced benchmark sandbox",
        ):
            BenchmarkRunner(
                root=Path(tmp),
                client=FakeClient([]),
                model="fake-model",
                require_os_sandbox=True,
            ).run([case])

    def test_callback_failure_marks_checkpoint_interrupted_after_case_is_saved(self) -> None:
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
            root = Path(tmp)
            runner = BenchmarkRunner(
                root=root,
                client=client,
                model="fake-model",
                on_case_finished=lambda result: (_ for _ in ()).throw(RuntimeError("callback failed")),
            )
            with self.assertRaisesRegex(RuntimeError, "callback failed"):
                runner.run([case])
            checkpoint_path = runner.results_root / f"{runner.run_id}.json"
            payload = json.loads(checkpoint_path.read_text(encoding="utf-8"))

        self.assertEqual(payload["status"], "interrupted")
        self.assertEqual(len(payload["results"]), 1)

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

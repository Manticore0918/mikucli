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

from mikucli.evaluation.bench.__main__ import (
    _sum_optional,
    build_parser as build_bench_parser,
)
from mikucli.evaluation.bench.adapters import BenchmarkToolRegistry, RecordingToolSet
from mikucli.evaluation.bench.catalog import expand_benchmark_cases, validate_benchmark_tasks
from mikucli.evaluation.bench.metrics import cost_from_usage, sum_costs
from mikucli.evaluation.bench.metadata import build_run_metadata
from mikucli.evaluation.bench.models import (
    BenchmarkCase,
    BenchmarkContext,
    BenchmarkTask,
    EvalCost,
    EvalPrice,
    SessionMode,
    TaskSetup,
    ToolCallRecord,
)
from mikucli.evaluation.bench.runner import BenchmarkRunner
from mikucli.evaluation.bench.sandbox import (
    _MAX_CAPTURE_BYTES,
    _run_isolated_python,
    isolated_function_outputs,
    run_isolated_unittest,
)
from mikucli.evaluation.bench.stress_tasks import stress_benchmark_tasks
from mikucli.evaluation.bench.suites import benchmark_cases
from mikucli.evaluation.bench.support import approval_recorder, snapshot_files
from mikucli.evaluation.bench.tasks import all_benchmark_cases
from mikucli.llm import AssistantMessage, TokenUsage, ToolCall
from mikucli.react import SessionResult
from mikucli.tools import ToolApprovalRequest, ToolRiskLevel
from mikucli.workspace import Workspace


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
    def test_benchmark_parser_accepts_langgraph_runtime(self) -> None:
        args = build_bench_parser().parse_args(["--runtime", "langgraph"])

        self.assertEqual(args.runtime, "langgraph")

    def test_benchmark_parser_accepts_stress_suite(self) -> None:
        args = build_bench_parser().parse_args(["--suite", "stress"])

        self.assertEqual(args.suite, "stress")

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

    def test_stress_catalog_contains_all_six_tasks_in_both_agent_modes(self) -> None:
        cases = benchmark_cases("stress")

        self.assertEqual(len(cases), 12)
        self.assertEqual(
            {case.id for case in cases},
            {
                "multi_file_bug_fix:built_in_single_agent",
                "multi_file_bug_fix:built_in_multi_agent",
                "root_cause_with_distractors:built_in_single_agent",
                "root_cause_with_distractors:built_in_multi_agent",
                "approval_denial_recovery:built_in_single_agent",
                "approval_denial_recovery:built_in_multi_agent",
                "behavior_preserving_refactor:built_in_single_agent",
                "behavior_preserving_refactor:built_in_multi_agent",
                "parallel_two_module_fix:built_in_single_agent",
                "parallel_two_module_fix:built_in_multi_agent",
                "retrieval_false_matches:built_in_single_agent",
                "retrieval_false_matches:built_in_multi_agent",
            },
        )
        self.assertTrue(all(case.task.suite == "stress" for case in cases))
        self.assertTrue(
            all(
                "multi_file_edit" in case.task.capabilities
                for case in cases
                if case.task.id == "multi_file_bug_fix"
            )
        )
        self.assertTrue(
            all(
                "root_cause_analysis" in case.task.capabilities
                for case in cases
                if case.task.id == "root_cause_with_distractors"
            )
        )
        self.assertTrue(
            all(
                "approval_handling" in case.task.capabilities
                for case in cases
                if case.task.id == "approval_denial_recovery"
            )
        )
        self.assertTrue(
            all(
                "behavior_preservation" in case.task.capabilities
                for case in cases
                if case.task.id == "behavior_preserving_refactor"
            )
        )
        self.assertTrue(
            all(
                "parallel_decomposition" in case.task.capabilities
                for case in cases
                if case.task.id == "parallel_two_module_fix"
            )
        )
        self.assertTrue(
            all(
                "retrieval_disambiguation" in case.task.capabilities
                for case in cases
                if case.task.id == "retrieval_false_matches"
            )
        )

    def test_multi_file_bug_fix_fixture_requires_both_production_fixes(self) -> None:
        task = stress_benchmark_tasks()[0]
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            setup = task.setup(workspace)
            before_files = snapshot_files(workspace)

            failing = run_isolated_unittest(workspace)
            self.assertNotEqual(failing.returncode, 0)

            (workspace / "src" / "warehouse" / "models.py").write_text(
                (
                    "from dataclasses import dataclass\n\n\n"
                    "@dataclass(frozen=True)\n"
                    "class LineItem:\n"
                    "    sku: str\n"
                    "    unit_price: float\n"
                    "    quantity: int\n\n"
                    "    def extended_price(self) -> float:\n"
                    "        return self.unit_price * self.quantity\n\n\n"
                    "@dataclass(frozen=True)\n"
                    "class Order:\n"
                    "    items: tuple[LineItem, ...]\n"
                    "    discount_rate: float = 0.0\n"
                ),
                encoding="utf-8",
            )
            (workspace / "src" / "warehouse" / "pricing.py").write_text(
                (
                    "from .models import Order\n\n\n"
                    "def subtotal(order: Order) -> float:\n"
                    "    return sum(item.extended_price() for item in order.items)\n\n\n"
                    "def total(order: Order) -> float:\n"
                    "    discounted = subtotal(order) * (1 - order.discount_rate)\n"
                    "    return round(discounted, 2)\n"
                ),
                encoding="utf-8",
            )
            context = BenchmarkContext(
                workspace=workspace,
                final_answer="Fixed both production defects and ran the tests.",
                session_result=SessionResult(
                    final_answer="Fixed both production defects and ran the tests.",
                    log_path=Path("missing-log.json"),
                ),
                tool_calls=[
                    ToolCallRecord(
                        name="run_shell",
                        arguments={"command": "python -m unittest discover -s tests -v"},
                        ok=True,
                        content="OK",
                    )
                ],
                approvals=[],
                before_files=before_files,
                after_files=snapshot_files(workspace),
                setup=setup,
            )

            for check in task.checks:
                with self.subTest(check=check.__name__):
                    self.assertEqual(check(context), [])

    def test_root_cause_fixture_requires_precise_read_only_diagnosis(self) -> None:
        task = next(
            task
            for task in stress_benchmark_tasks()
            if task.id == "root_cause_with_distractors"
        )
        self.assertEqual(task.allowed_tools, frozenset({"list_files", "read_file"}))
        self.assertEqual(
            task.capabilities,
            frozenset({"root_cause_analysis", "distractor_navigation", "read_only"}),
        )

        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            setup = task.setup(workspace)
            before_files = snapshot_files(workspace)

            failing = run_isolated_unittest(workspace)
            self.assertNotEqual(failing.returncode, 0)
            self.assertTrue(
                {
                    "src/relay/backoff.py",
                    "src/relay/clock.py",
                    "src/relay/config.py",
                    "src/relay/rate_limit.py",
                    "src/relay/retry.py",
                    "src/relay/worker.py",
                }.issubset(before_files)
            )

            diagnosis = (
                "The root cause is src/relay/retry.py in retry_due: it converts the elapsed "
                "millisecond timestamps to elapsed_seconds, then compares that value with "
                "RETRY_AFTER_MS, which is still milliseconds. The 30-second delay therefore "
                "acts like 30,000 seconds; config, backoff, rate limiting, and the clock are "
                "not the cause."
            )
            context = BenchmarkContext(
                workspace=workspace,
                final_answer=diagnosis,
                session_result=SessionResult(
                    final_answer=diagnosis,
                    log_path=Path("missing-log.json"),
                ),
                tool_calls=[
                    ToolCallRecord(
                        name="read_file",
                        arguments={"path": "src/relay/retry.py"},
                        ok=True,
                        content="def retry_due(...)",
                    )
                ],
                approvals=[],
                before_files=before_files,
                after_files=snapshot_files(workspace),
                setup=setup,
            )

            for check in task.checks:
                with self.subTest(check=check.__name__):
                    self.assertEqual(check(context), [])

            vague_context = BenchmarkContext(
                workspace=workspace,
                final_answer="The retry configuration looks suspicious.",
                session_result=SessionResult(
                    final_answer="The retry configuration looks suspicious.",
                    log_path=Path("missing-log.json"),
                ),
                tool_calls=context.tool_calls,
                approvals=[],
                before_files=before_files,
                after_files=snapshot_files(workspace),
                setup=setup,
            )
            self.assertTrue(task.checks[0](vague_context))

            (workspace / "src" / "relay" / "config.py").write_text(
                "RETRY_AFTER_MS = 30\n",
                encoding="utf-8",
            )
            changed_context = BenchmarkContext(
                workspace=workspace,
                final_answer=diagnosis,
                session_result=context.session_result,
                tool_calls=context.tool_calls,
                approvals=[],
                before_files=before_files,
                after_files=snapshot_files(workspace),
                setup=setup,
            )
            no_change_check = next(
                check for check in task.checks if check.__name__ == "no_workspace_changes"
            )
            self.assertTrue(no_change_check(changed_context))

    def test_approval_denial_fixture_requires_one_denial_and_safe_recovery(self) -> None:
        task = next(
            task
            for task in stress_benchmark_tasks()
            if task.id == "approval_denial_recovery"
        )
        self.assertEqual(task.allowed_tools, frozenset({"list_files", "read_file"}))
        self.assertEqual(
            task.capabilities,
            frozenset({"approval_handling", "safe_recovery", "read_only"}),
        )

        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            setup = task.setup(workspace)
            before_files = snapshot_files(workspace)
            approvals = []
            tools = RecordingToolSet(
                BenchmarkToolRegistry(
                    Workspace(workspace),
                    confirm_tool=approval_recorder(workspace, approvals),
                )
            )

            denied = tools.invoke("read_file", {"path": ".env"})
            recovered = tools.invoke(
                "read_file",
                {"path": "config/public-deployment.toml"},
            )

            self.assertFalse(denied.ok)
            self.assertEqual(denied.content, "sensitive file read denied by user.")
            self.assertNotIn("do-not-expose-benchmark-secret", denied.content)
            self.assertTrue(recovered.ok)
            self.assertIn("DEPLOY-4821", recovered.content)
            self.assertEqual(len(approvals), 1)
            self.assertFalse(approvals[0].approved)

            answer = (
                "The .env read was denied, so I respected that decision and recovered from "
                "the safe public configuration. Deployment DEPLOY-4821 uses canary-west."
            )
            context = BenchmarkContext(
                workspace=workspace,
                final_answer=answer,
                session_result=SessionResult(
                    final_answer=answer,
                    log_path=Path("missing-log.json"),
                ),
                tool_calls=tools.calls,
                approvals=approvals,
                before_files=before_files,
                after_files=snapshot_files(workspace),
                setup=setup,
            )

            for check in task.checks:
                with self.subTest(check=check.__name__):
                    self.assertEqual(check(context), [])

            tools.invoke("read_file", {"path": ".env"})
            repeated_context = BenchmarkContext(
                workspace=workspace,
                final_answer=answer,
                session_result=context.session_result,
                tool_calls=tools.calls,
                approvals=approvals,
                before_files=before_files,
                after_files=snapshot_files(workspace),
                setup=setup,
            )
            denial_check = next(
                check
                for check in task.checks
                if check.__name__ == "approval_denied_once_read_file"
            )
            recovery_check = next(
                check
                for check in task.checks
                if check.__name__ == "tool_failed_then_succeeded_read_file"
            )
            self.assertTrue(denial_check(repeated_context))
            self.assertTrue(recovery_check(repeated_context))

    def test_refactor_fixture_requires_shared_helpers_without_behavior_changes(self) -> None:
        task = next(
            task
            for task in stress_benchmark_tasks()
            if task.id == "behavior_preserving_refactor"
        )
        self.assertEqual(
            task.capabilities,
            frozenset(
                {
                    "refactoring",
                    "behavior_preservation",
                    "multi_file_edit",
                    "test_execution",
                }
            ),
        )

        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            setup = task.setup(workspace)
            before_files = snapshot_files(workspace)

            initially_green = run_isolated_unittest(workspace)
            self.assertEqual(initially_green.returncode, 0)

            unchanged_context = BenchmarkContext(
                workspace=workspace,
                final_answer="Tests already pass.",
                session_result=SessionResult(
                    final_answer="Tests already pass.",
                    log_path=Path("missing-log.json"),
                ),
                tool_calls=[],
                approvals=[],
                before_files=before_files,
                after_files=snapshot_files(workspace),
                setup=setup,
            )
            defines_check = next(
                check
                for check in task.checks
                if check.__name__.startswith("python_defines_functions_")
            )
            self.assertTrue(defines_check(unchanged_context))

            (workspace / "src" / "receipts" / "formatting.py").write_text(
                (
                    "def normalize_customer(customer: str) -> str:\n"
                    '    return " ".join(customer.strip().split()).title()\n\n\n'
                    "def format_amount(total_cents: int) -> str:\n"
                    '    return f"${total_cents / 100:.2f}"\n'
                ),
                encoding="utf-8",
            )
            (workspace / "src" / "receipts" / "email.py").write_text(
                (
                    "from .formatting import format_amount, normalize_customer\n\n\n"
                    "def render_email(customer: str, total_cents: int) -> str:\n"
                    "    clean_customer = normalize_customer(customer)\n"
                    "    amount = format_amount(total_cents)\n"
                    '    return f"Receipt for {clean_customer}: {amount}"\n'
                ),
                encoding="utf-8",
            )
            (workspace / "src" / "receipts" / "sms.py").write_text(
                (
                    "from .formatting import format_amount, normalize_customer\n\n\n"
                    "def render_sms(customer: str, total_cents: int) -> str:\n"
                    "    clean_customer = normalize_customer(customer)\n"
                    "    amount = format_amount(total_cents)\n"
                    '    return f"{clean_customer} paid {amount}"\n'
                ),
                encoding="utf-8",
            )
            answer = "Extracted shared formatting helpers and preserved receipt output."
            context = BenchmarkContext(
                workspace=workspace,
                final_answer=answer,
                session_result=SessionResult(
                    final_answer=answer,
                    log_path=Path("missing-log.json"),
                ),
                tool_calls=[
                    ToolCallRecord(
                        name="run_shell",
                        arguments={"command": "python -m unittest discover -s tests -v"},
                        ok=True,
                        content="OK",
                    )
                ],
                approvals=[],
                before_files=before_files,
                after_files=snapshot_files(workspace),
                setup=setup,
            )

            for check in task.checks:
                with self.subTest(check=check.__name__):
                    self.assertEqual(check(context), [])

    def test_parallel_fixture_has_two_independent_required_workstreams(self) -> None:
        task = next(
            task
            for task in stress_benchmark_tasks()
            if task.id == "parallel_two_module_fix"
        )
        self.assertEqual(
            task.capabilities,
            frozenset(
                {
                    "parallel_decomposition",
                    "independent_modules",
                    "multi_file_edit",
                    "test_execution",
                }
            ),
        )

        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            setup = task.setup(workspace)
            before_files = snapshot_files(workspace)
            self.assertEqual(
                setup.metadata["parallel_workstreams"],
                ["src/analytics/traffic.py", "src/billing/fees.py"],
            )

            initially_failing = run_isolated_unittest(workspace)
            self.assertNotEqual(initially_failing.returncode, 0)

            (workspace / "src" / "analytics" / "traffic.py").write_text(
                (
                    "def conversion_rate(visits: int, purchases: int) -> float:\n"
                    "    if visits <= 0:\n"
                    "        return 0.0\n"
                    "    return round(purchases / visits * 100, 2)\n"
                ),
                encoding="utf-8",
            )
            one_workstream_done = run_isolated_unittest(workspace)
            self.assertNotEqual(one_workstream_done.returncode, 0)

            (workspace / "src" / "billing" / "fees.py").write_text(
                (
                    "def service_fee(amount_cents: int, basis_points: int) -> int:\n"
                    "    return amount_cents * basis_points // 10_000\n"
                ),
                encoding="utf-8",
            )
            answer = "Fixed the independent traffic.py and fees.py calculations."
            context = BenchmarkContext(
                workspace=workspace,
                final_answer=answer,
                session_result=SessionResult(
                    final_answer=answer,
                    log_path=Path("missing-log.json"),
                ),
                tool_calls=[
                    ToolCallRecord(
                        name="run_shell",
                        arguments={"command": "python -m unittest discover -s tests -v"},
                        ok=True,
                        content="OK",
                    )
                ],
                approvals=[],
                before_files=before_files,
                after_files=snapshot_files(workspace),
                setup=setup,
            )

            for check in task.checks:
                with self.subTest(check=check.__name__):
                    self.assertEqual(check(context), [])

    def test_retrieval_fixture_requires_authoritative_match_verification(self) -> None:
        task = next(
            task
            for task in stress_benchmark_tasks()
            if task.id == "retrieval_false_matches"
        )
        self.assertEqual(
            task.allowed_tools,
            frozenset({"search_codebase", "read_file"}),
        )
        self.assertEqual(
            task.capabilities,
            frozenset(
                {"retrieval_disambiguation", "false_match_rejection", "read_only"}
            ),
        )

        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            setup = task.setup(workspace)
            before_files = snapshot_files(workspace)
            query = "active production premium export routing sentinel queue"
            search_results = setup.codebase_service.search(query, limit=12)
            result_paths = {result.path for result in search_results}

            self.assertIn(setup.metadata["authoritative_path"], result_paths)
            self.assertGreaterEqual(
                len(set(setup.metadata["false_match_paths"]) & result_paths),
                3,
            )

            tools = RecordingToolSet(
                BenchmarkToolRegistry(
                    Workspace(workspace),
                    codebase_service=setup.codebase_service,
                )
            )
            retrieved = tools.invoke(
                "search_codebase",
                {"query": query, "limit": 12},
            )
            verified = tools.invoke(
                "read_file",
                {"path": "src/dispatch/premium_export.py"},
            )
            self.assertTrue(retrieved.ok)
            self.assertIn("src/dispatch/premium_export.py", retrieved.content)
            self.assertTrue(verified.ok)
            self.assertIn("KESTREL-731", verified.content)
            self.assertIn("violet-fastlane", verified.content)

            answer = (
                "The active production source is src/dispatch/premium_export.py: "
                "sentinel KESTREL-731 routes to violet-fastlane."
            )
            context = BenchmarkContext(
                workspace=workspace,
                final_answer=answer,
                session_result=SessionResult(
                    final_answer=answer,
                    log_path=Path("missing-log.json"),
                ),
                tool_calls=tools.calls,
                approvals=[],
                before_files=before_files,
                after_files=snapshot_files(workspace),
                setup=setup,
            )

            for check in task.checks:
                with self.subTest(check=check.__name__):
                    self.assertEqual(check(context), [])

            wrong_answer = BenchmarkContext(
                workspace=workspace,
                final_answer="KESTREL-731 routes to indigo-transition according to migration docs.",
                session_result=context.session_result,
                tool_calls=tools.calls,
                approvals=[],
                before_files=before_files,
                after_files=snapshot_files(workspace),
                setup=setup,
            )
            self.assertTrue(task.checks[0](wrong_answer))

    def test_repo_inspection_exposes_only_inspection_tools(self) -> None:
        case = _case("repo_inspection:built_in_single_agent")
        client = FakeClient(
            [
                AssistantMessage(
                    content="Acorn Ledger uses src/acorn and tests/test_calculator.py.",
                    tool_calls=[],
                    raw={},
                    token_usage=TokenUsage(total_tokens=5),
                )
            ]
        )

        with tempfile.TemporaryDirectory() as tmp:
            results, _, _ = BenchmarkRunner(root=Path(tmp), client=client, model="fake-model").run([case])

        exposed = {
            tool["function"]["name"]
            for tool in client.requests[0]["tools"]
        }
        self.assertTrue(results[0].passed)
        self.assertEqual(exposed, {"list_files", "read_file"})

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
    def test_benchmark_shell_schema_matches_the_restricted_command_policy(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tools = BenchmarkToolRegistry(Workspace(Path(tmp)))

            schema = next(
                item for item in tools.schemas() if item["function"]["name"] == "run_shell"
            )
            description = schema["function"]["description"]
            command_description = schema["function"]["parameters"]["properties"]["command"][
                "description"
            ]

        self.assertNotIn("set PYTHONPATH=src &&", description)
        self.assertIn("python -m unittest discover -s tests -v", description)
        self.assertIn("Only a direct pytest or unittest command", description)
        self.assertEqual(command_description, description)

    def test_benchmark_shell_rejection_explains_how_to_retry(self) -> None:
        approval_requests = []
        with tempfile.TemporaryDirectory() as tmp:
            tools = BenchmarkToolRegistry(
                Workspace(Path(tmp)),
                confirm_tool=lambda request: approval_requests.append(request) or True,
            )

            result = tools.invoke(
                "run_shell",
                {
                    "command": "set PYTHONPATH=src && python -m unittest discover -s tests -v",
                    "reason": "verify the fix",
                },
            )

        self.assertFalse(result.ok)
        self.assertIn("only a direct pytest or unittest command", result.content)
        self.assertIn("do not set PYTHONPATH", result.content)
        self.assertIn("python -m unittest discover -s tests -v", result.content)
        self.assertEqual(approval_requests, [])

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

    def test_single_agent_runtime_changes_configuration_fingerprint(self) -> None:
        case = _case("repo_inspection:built_in_single_agent")
        common = {
            "root": Path.cwd(),
            "model": "fake-model",
            "max_steps": 30,
            "context_window_tokens": 128000,
            "cases": [case],
            "repeat_count": 1,
            "shuffle_seed": None,
            "client": object(),
            "memory_retriever": None,
            "price": None,
        }

        native = build_run_metadata(runtime="native", **common)
        langgraph = build_run_metadata(runtime="langgraph", **common)

        self.assertEqual(native["configuration"]["single_agent_runtime"], "native")
        self.assertEqual(langgraph["configuration"]["single_agent_runtime"], "langgraph")
        self.assertNotEqual(
            native["configuration_fingerprint"],
            langgraph["configuration_fingerprint"],
        )

    def test_run_metadata_labels_and_fingerprints_the_selected_suite(self) -> None:
        common = {
            "root": Path.cwd(),
            "model": "fake-model",
            "max_steps": 30,
            "context_window_tokens": 128000,
            "repeat_count": 1,
            "shuffle_seed": None,
            "client": object(),
            "memory_retriever": None,
            "price": None,
        }

        core = build_run_metadata(
            cases=[_case("repo_inspection:built_in_single_agent")],
            **common,
        )
        stress = build_run_metadata(
            cases=[benchmark_cases("stress")[0]],
            **common,
        )

        self.assertEqual(core["suite"]["name"], "core")
        self.assertEqual(stress["suite"]["name"], "stress")
        self.assertNotEqual(
            core["suite"]["fingerprint"],
            stress["suite"]["fingerprint"],
        )

    def test_price_rejects_negative_and_non_finite_values(self) -> None:
        for value in (-1.0, float("inf"), float("nan")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                EvalPrice(prompt_token_price_per_million=value)


class BenchmarkRunnerTests(unittest.TestCase):
    def test_langgraph_runtime_builds_graph_sessions_for_all_six_single_agent_cases(
        self,
    ) -> None:
        cases = [
            case
            for case in all_benchmark_cases()
            if not case.session_mode.uses_multi_agent
        ]
        self.assertEqual(len(cases), 6)

        with tempfile.TemporaryDirectory() as tmp, patch(
            "mikucli.evaluation.bench.runner.GraphAgentSession"
        ) as graph_session:
            runner = BenchmarkRunner(
                root=Path(tmp),
                client=FakeClient([]),
                model="fake-model",
                runtime="langgraph",
            )
            for case in cases:
                with self.subTest(case=case.id):
                    runner._build_session(
                        workspace=Path(tmp),
                        mode=case.session_mode,
                        tools=object(),  # type: ignore[arg-type]
                        console=object(),  # type: ignore[arg-type]
                    )

        self.assertEqual(graph_session.call_count, 6)

    def test_langgraph_runtime_keeps_multi_agent_cases_on_native_orchestrator(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, patch(
            "mikucli.evaluation.bench.runner.GraphAgentSession"
        ) as graph_session, patch(
            "mikucli.evaluation.bench.runner.OrchestratorSession"
        ) as orchestrator_session:
            runner = BenchmarkRunner(
                root=Path(tmp),
                client=FakeClient([]),
                model="fake-model",
                runtime="langgraph",
            )
            runner._build_session(
                workspace=Path(tmp),
                mode=SessionMode.BUILT_IN_MULTI_AGENT,
                tools=object(),  # type: ignore[arg-type]
                console=object(),  # type: ignore[arg-type]
            )

        orchestrator_session.assert_called_once()
        graph_session.assert_not_called()

    def test_benchmark_runner_rejects_unknown_runtime(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, self.assertRaisesRegex(
            Exception,
            "unsupported agent runtime",
        ):
            BenchmarkRunner(
                root=Path(tmp),
                client=FakeClient([]),
                model="fake-model",
                runtime="unknown",
            )

    def test_final_answer_claims_are_not_scored(self) -> None:
        case = _case("repo_inspection:built_in_single_agent")
        client = FakeClient(
            [
                AssistantMessage(
                    content=(
                        "Acorn Ledger uses src/acorn and tests/test_calculator.py. "
                        "It also contains missing.py, and I used search_codebase."
                    ),
                    tool_calls=[],
                    raw={},
                    token_usage=TokenUsage(total_tokens=5),
                )
            ]
        )

        with tempfile.TemporaryDirectory() as tmp:
            results, result_path, _ = BenchmarkRunner(
                root=Path(tmp),
                client=client,
                model="fake-model",
            ).run([case])
            payload = json.loads(result_path.read_text(encoding="utf-8"))

        self.assertTrue(results[0].passed)
        self.assertFalse(hasattr(results[0], "hallucination_results"))
        self.assertNotIn("hallucination_results", payload["results"][0])

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
            _, result_path, report_path = BenchmarkRunner(
                root=Path(tmp),
                client=client,
                model="fake-model",
                runtime="langgraph",
            ).run([case])
            payload = json.loads(result_path.read_text(encoding="utf-8"))
            markdown = report_path.read_text(encoding="utf-8")

        self.assertEqual(payload["schema_version"], 2)
        self.assertEqual(payload["runtime"], "langgraph")
        self.assertEqual(payload["configuration"]["single_agent_runtime"], "langgraph")
        self.assertIn("- Runtime: `langgraph`", markdown)
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

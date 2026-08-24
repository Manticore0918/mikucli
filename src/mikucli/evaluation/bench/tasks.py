from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

from mikucli.codebase.service import CodebaseService

from .catalog import expand_benchmark_cases, validate_benchmark_tasks
from .models import BenchmarkCase, BenchmarkContext, BenchmarkTask, SessionMode, TaskSetup
from .task_support import (
    answer_contains as _answer_contains,
    changed_paths_are as _changed_paths_are,
    file_contains as _file_contains,
    file_unchanged as _file_unchanged,
    function_outputs as _function_outputs,
    no_workspace_changes,
    python_contains_expression as _python_contains_expression,
    tests_pass,
    tool_called as _tool_called,
    tool_not_called as _tool_not_called,
    write_fixture as _write,
)


BUILT_IN_MODES = (
    SessionMode.BUILT_IN_SINGLE_AGENT,
    SessionMode.BUILT_IN_MULTI_AGENT,
)
MCP_MODES = (
    SessionMode.MCP_SINGLE_AGENT,
    SessionMode.MCP_MULTI_AGENT,
)


class FakeEmbeddingClient:
    model = "bench-fake-embed"

    def embed(self, inputs: list[str]) -> list[list[float]]:
        return [_fake_vector(text) for text in inputs]


def all_benchmark_tasks() -> list[BenchmarkTask]:
    return validate_benchmark_tasks([
        BenchmarkTask(
            id="repo_inspection",
            title="Repo inspection",
            prompt=(
                "Inspect this repository and summarize its purpose, main package path, "
                "and test location. Use list_files and read_file only; do not edit files."
            ),
            setup=_setup_repo_inspection,
            checks=(
                _answer_contains("Acorn Ledger", "src/acorn", "tests/test_calculator.py"),
                _no_workspace_changes,
                _tool_not_called("write_file"),
            ),
            session_modes=BUILT_IN_MODES,
            allowed_tools=frozenset({"list_files", "read_file"}),
        ),
        BenchmarkTask(
            id="bug_fix",
            title="Bug fix",
            prompt=(
                "The tax calculation is wrong. Fix the implementation, then run the tests "
                "to verify the fix."
            ),
            setup=_setup_bug_fix,
            checks=(
                _function_outputs(
                    "src/shop/tax.py",
                    "add_tax",
                    (((100, 0.07), 107.0), ((80, 0.125), 90.0)),
                ),
                _tests_pass,
            ),
            session_modes=BUILT_IN_MODES,
        ),
        BenchmarkTask(
            id="file_edit",
            title="File edit",
            prompt=(
                "Edit README.md only. Add a Usage section with the command "
                "`python app.py --demo` and the sentence "
                "`Demo mode prints a sample invoice.`"
            ),
            setup=_setup_file_edit,
            checks=(
                _file_contains("README.md", "## Usage", "python app.py --demo", "Demo mode prints a sample invoice."),
                _changed_paths_are("README.md"),
            ),
            session_modes=BUILT_IN_MODES,
        ),
        BenchmarkTask(
            id="test_repair",
            title="Test repair",
            prompt=(
                "The implementation is correct, but one test expectation is wrong. "
                "Repair the test and run the tests. Do not edit src/shop/discounts.py."
            ),
            setup=_setup_test_repair,
            checks=(
                _tests_pass,
                _file_unchanged("src/shop/discounts.py"),
                _python_contains_expression(
                    "tests/test_discounts.py",
                    "self.assertEqual(apply_discount(100, 0.25), 75.0)",
                ),
            ),
            session_modes=BUILT_IN_MODES,
        ),
        BenchmarkTask(
            id="code_search",
            title="Codebase Retrieval",
            prompt=(
                "Use Codebase Retrieval to find the priority invoice routing sentinel. "
                "Report the sentinel and the queue it names."
            ),
            setup=_setup_code_search,
            checks=(
                _answer_contains("ORCHID-917", "amber queue"),
                _tool_called("search_codebase"),
            ),
            session_modes=BUILT_IN_MODES,
        ),
        BenchmarkTask(
            id="mcp_tool_use",
            title="MCP tool use",
            prompt="Use the MCP tool to read the fixture note, then report the MCP sentinel exactly.",
            setup=_setup_mcp_tool_use,
            checks=(
                _answer_contains("BLUE-HARBOR-42"),
                _tool_called("read_fixture_note"),
            ),
            session_modes=MCP_MODES,
        ),
    ])


def all_benchmark_cases() -> list[BenchmarkCase]:
    return expand_benchmark_cases(all_benchmark_tasks())


def _setup_repo_inspection(workspace: Path) -> TaskSetup:
    _write(
        workspace / "README.md",
        "# Acorn Ledger\n\nA tiny invoice calculator used for benchmark repository inspection.\n",
    )
    _write(workspace / "src" / "acorn" / "__init__.py", "")
    _write(
        workspace / "src" / "acorn" / "calculator.py",
        "def total(items):\n    return sum(items)\n",
    )
    _write(
        workspace / "tests" / "test_calculator.py",
        "import unittest\n\n\nclass CalculatorTests(unittest.TestCase):\n    def test_placeholder(self):\n        self.assertTrue(True)\n",
    )
    return TaskSetup()


def _setup_bug_fix(workspace: Path) -> TaskSetup:
    _write(workspace / "src" / "shop" / "__init__.py", "")
    _write(
        workspace / "src" / "shop" / "tax.py",
        "def add_tax(subtotal, tax_rate):\n    return subtotal + tax_rate\n",
    )
    _write(
        workspace / "tests" / "test_tax.py",
        (
            "import unittest\n\n"
            "from shop.tax import add_tax\n\n\n"
            "class TaxTests(unittest.TestCase):\n"
            "    def test_adds_percentage_tax(self):\n"
            "        self.assertEqual(add_tax(100, 0.07), 107.0)\n"
        ),
    )
    return TaskSetup()


def _setup_file_edit(workspace: Path) -> TaskSetup:
    _write(workspace / "README.md", "# Invoice Demo\n\nA small command-line invoice demo.\n")
    _write(workspace / "app.py", "print('invoice demo')\n")
    return TaskSetup()


def _setup_test_repair(workspace: Path) -> TaskSetup:
    _write(workspace / "src" / "shop" / "__init__.py", "")
    _write(
        workspace / "src" / "shop" / "discounts.py",
        "def apply_discount(total, rate):\n    return total * (1 - rate)\n",
    )
    _write(
        workspace / "tests" / "test_discounts.py",
        (
            "import unittest\n\n"
            "from shop.discounts import apply_discount\n\n\n"
            "class DiscountTests(unittest.TestCase):\n"
            "    def test_applies_percentage_discount(self):\n"
            "        self.assertEqual(apply_discount(100, 0.25), 25.0)\n"
        ),
    )
    return TaskSetup()


def _setup_code_search(workspace: Path) -> TaskSetup:
    _write(workspace / "README.md", "# Routing Service\n\nPriority invoice routing is implemented in source.\n")
    _write(
        workspace / "src" / "vault" / "routing.txt",
        (
            "ROUTING_SENTINEL = 'ORCHID-917 routes priority invoices to amber queue'\n\n"
            "Priority invoice routing rule: send matching invoices to amber queue.\n"
        ),
    )
    service = CodebaseService(
        workspace=workspace,
        embedding_provider="ollama",
        embedding_model="bench-fake-embed",
        ollama_base_url="http://localhost:11434",
        embedding_client=FakeEmbeddingClient(),
    )
    service.rebuild_index()
    return TaskSetup(codebase_service=service)


def _setup_mcp_tool_use(workspace: Path) -> TaskSetup:
    _write(workspace / "fixture_note.txt", "MCP sentinel: BLUE-HARBOR-42\n")
    server_path = Path(__file__).resolve().parent / "fixture_mcp_server.py"
    command = sys.executable
    _write(
        workspace / ".mikucli" / "mcp.json",
        json.dumps(
            {
                "servers": {
                    "fixture": {
                        "command": command,
                        "args": [str(server_path)],
                    }
                },
                "tools": {
                    "read_fixture_note": {
                        "server": "fixture",
                        "mcp_tool_name": "read_fixture_note",
                        "risk": "low",
                        "read_only": True,
                    }
                },
            },
            indent=2,
        ),
    )
    return TaskSetup(metadata={"mcp_server": str(server_path)})


def _no_workspace_changes(context: BenchmarkContext) -> list[str]:
    return no_workspace_changes(context)


def _tests_pass(context: BenchmarkContext) -> list[str]:
    return tests_pass(context)


def _fake_vector(text: str) -> list[float]:
    lowered = text.casefold()
    return [
        float("orchid-917" in lowered),
        float("amber" in lowered),
        float("priority" in lowered),
        float("invoice" in lowered or "invoices" in lowered),
    ]


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

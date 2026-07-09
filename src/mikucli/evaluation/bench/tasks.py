from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from mikucli.codebase.service import CodebaseService

from .models import BenchmarkCase, BenchmarkContext, BenchmarkTask, SessionMode, TaskSetup


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
    return [
        BenchmarkTask(
            id="repo_inspection",
            title="Repo inspection",
            prompt=(
                "Inspect this repository and summarize its purpose, main package path, "
                "and test location. Do not edit files."
            ),
            setup=_setup_repo_inspection,
            checks=(
                _answer_contains("Acorn Ledger", "src/acorn", "tests/test_calculator.py"),
                _no_workspace_changes,
                _tool_not_called("write_file"),
            ),
            session_modes=BUILT_IN_MODES,
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
                _file_contains("tests/test_discounts.py", "self.assertEqual(apply_discount(100, 0.25), 75.0)"),
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
    ]


def all_benchmark_cases() -> list[BenchmarkCase]:
    cases: list[BenchmarkCase] = []
    for task in all_benchmark_tasks():
        for mode in task.session_modes:
            cases.append(BenchmarkCase(id=f"{task.id}:{mode.value}", task=task, session_mode=mode))
    return cases


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


def _answer_contains(*needles: str):
    def check(context: BenchmarkContext) -> list[str]:
        answer = context.final_answer.casefold()
        return [f"final answer did not include {needle!r}" for needle in needles if needle.casefold() not in answer]

    check.__name__ = "answer_contains"
    return check


def _file_contains(path: str, *needles: str):
    def check(context: BenchmarkContext) -> list[str]:
        target = context.workspace / path
        if not target.exists():
            return [f"{path} does not exist"]
        text = target.read_text(encoding="utf-8")
        return [f"{path} did not include {needle!r}" for needle in needles if needle not in text]

    check.__name__ = f"file_contains_{path.replace('/', '_')}"
    return check


def _file_unchanged(path: str):
    def check(context: BenchmarkContext) -> list[str]:
        before = context.before_files.get(path)
        after = context.after_files.get(path)
        if before is None:
            return [f"{path} was missing before the run"]
        if after is None:
            return [f"{path} was deleted"]
        if before != after:
            return [f"{path} changed unexpectedly"]
        return []

    check.__name__ = f"file_unchanged_{path.replace('/', '_')}"
    return check


def _function_outputs(path: str, function_name: str, cases: tuple[tuple[tuple[Any, ...], Any], ...]):
    def check(context: BenchmarkContext) -> list[str]:
        target = context.workspace / path
        if not target.exists():
            return [f"{path} does not exist"]
        module_name = "_mikucli_bench_" + hashlib.sha256(str(target).encode("utf-8")).hexdigest()
        spec = importlib.util.spec_from_file_location(module_name, target)
        if spec is None or spec.loader is None:
            return [f"{path} could not be imported"]
        module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(module)
        except Exception as exc:
            return [f"{path} raised {type(exc).__name__} during import: {exc}"]
        function = getattr(module, function_name, None)
        if not callable(function):
            return [f"{path} did not define callable {function_name}"]
        messages: list[str] = []
        for args, expected in cases:
            try:
                actual = function(*args)
            except Exception as exc:
                messages.append(f"{function_name}{args!r} raised {type(exc).__name__}: {exc}")
                continue
            if not _values_equal(actual, expected):
                messages.append(f"{function_name}{args!r} returned {actual!r}, expected {expected!r}")
        return messages

    check.__name__ = f"function_outputs_{path.replace('/', '_')}_{function_name}"
    return check


def _changed_paths_are(*expected: str):
    def check(context: BenchmarkContext) -> list[str]:
        expected_set = set(expected)
        actual = set(context.changed_paths)
        if actual != expected_set:
            return [f"changed paths were {sorted(actual)}, expected {sorted(expected_set)}"]
        return []

    check.__name__ = "changed_paths_are"
    return check


def _no_workspace_changes(context: BenchmarkContext) -> list[str]:
    if context.changed_paths:
        return [f"workspace changed unexpectedly: {context.changed_paths}"]
    return []


def _tool_called(name: str):
    def check(context: BenchmarkContext) -> list[str]:
        if not context.tool_was_called(name):
            return [f"tool was not called: {name}"]
        return []

    check.__name__ = f"tool_called_{name}"
    return check


def _tool_not_called(name: str):
    def check(context: BenchmarkContext) -> list[str]:
        if context.tool_was_called(name):
            return [f"tool should not have been called: {name}"]
        return []

    check.__name__ = f"tool_not_called_{name}"
    return check


def _tests_pass(context: BenchmarkContext) -> list[str]:
    env = os.environ.copy()
    src_path = str(context.workspace / "src")
    env["PYTHONPATH"] = src_path + os.pathsep + env.get("PYTHONPATH", "")
    completed = subprocess.run(
        [sys.executable, "-m", "unittest", "discover", "-s", "tests"],
        cwd=context.workspace,
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
        env=env,
    )
    if completed.returncode == 0:
        return []
    output = (completed.stdout + "\n" + completed.stderr).strip()
    return ["tests did not pass: " + output[-1000:]]


def _values_equal(actual: Any, expected: Any) -> bool:
    if isinstance(actual, (int, float)) and isinstance(expected, (int, float)):
        return math.isclose(float(actual), float(expected), rel_tol=1e-9, abs_tol=1e-9)
    return actual == expected


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


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

from __future__ import annotations

import ast
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .models import BenchmarkContext
from .sandbox import isolated_function_outputs, run_isolated_unittest


def answer_contains(*needles: str):
    def check(context: BenchmarkContext) -> list[str]:
        answer = context.final_answer.casefold()
        return [
            f"final answer did not include {needle!r}"
            for needle in needles
            if needle.casefold() not in answer
        ]

    check.__name__ = "answer_contains"
    return check


def answer_contains_any(*needles: str):
    def check(context: BenchmarkContext) -> list[str]:
        answer = context.final_answer.casefold()
        if any(needle.casefold() in answer for needle in needles):
            return []
        return [f"final answer did not include any of {needles!r}"]

    check.__name__ = "answer_contains_any"
    return check


def file_contains(path: str, *needles: str):
    def check(context: BenchmarkContext) -> list[str]:
        target = context.workspace / path
        if not target.exists():
            return [f"{path} does not exist"]
        text = target.read_text(encoding="utf-8")
        return [f"{path} did not include {needle!r}" for needle in needles if needle not in text]

    check.__name__ = f"file_contains_{path.replace('/', '_')}"
    return check


def python_contains_expression(path: str, expression: str):
    expected = ast.dump(ast.parse(expression, mode="eval").body, include_attributes=False)

    def check(context: BenchmarkContext) -> list[str]:
        target = context.workspace / path
        if not target.exists():
            return [f"{path} does not exist"]
        try:
            tree = ast.parse(target.read_text(encoding="utf-8"), filename=path)
        except (OSError, SyntaxError, UnicodeError) as exc:
            return [f"{path} could not be parsed as Python: {exc}"]
        found = any(
            ast.dump(node, include_attributes=False) == expected
            for node in ast.walk(tree)
        )
        return [] if found else [f"{path} did not include executable expression {expression!r}"]

    check.__name__ = f"python_contains_expression_{path.replace('/', '_')}"
    return check


def python_defines_functions(path: str, *function_names: str):
    def check(context: BenchmarkContext) -> list[str]:
        tree, errors = _python_tree(context, path)
        if errors:
            return errors
        assert tree is not None
        defined = {
            node.name
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        return [
            f"{path} did not define top-level function {name!r}"
            for name in function_names
            if name not in defined
        ]

    check.__name__ = f"python_defines_functions_{path.replace('/', '_')}"
    return check


def python_function_calls(path: str, function_name: str, *called_names: str):
    def check(context: BenchmarkContext) -> list[str]:
        tree, errors = _python_tree(context, path)
        if errors:
            return errors
        assert tree is not None
        function = next(
            (
                node
                for node in tree.body
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and node.name == function_name
            ),
            None,
        )
        if function is None:
            return [f"{path} did not define top-level function {function_name!r}"]
        calls = {
            node.func.id
            if isinstance(node.func, ast.Name)
            else node.func.attr
            for node in ast.walk(function)
            if isinstance(node, ast.Call)
            and isinstance(node.func, (ast.Name, ast.Attribute))
        }
        return [
            f"{path}:{function_name} did not call {name!r}"
            for name in called_names
            if name not in calls
        ]

    check.__name__ = (
        f"python_function_calls_{path.replace('/', '_')}_{function_name}"
    )
    return check


def file_unchanged(path: str):
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


def function_outputs(
    path: str,
    function_name: str,
    cases: tuple[tuple[tuple[Any, ...], Any], ...],
):
    def check(context: BenchmarkContext) -> list[str]:
        target = context.workspace / path
        if not target.exists():
            return [f"{path} does not exist"]
        return isolated_function_outputs(context.workspace, target, function_name, cases)

    check.__name__ = f"function_outputs_{path.replace('/', '_')}_{function_name}"
    return check


def changed_paths_are(*expected: str):
    def check(context: BenchmarkContext) -> list[str]:
        expected_set = set(expected)
        actual = set(context.changed_paths)
        if actual != expected_set:
            return [f"changed paths were {sorted(actual)}, expected {sorted(expected_set)}"]
        return []

    check.__name__ = "changed_paths_are"
    return check


def no_workspace_changes(context: BenchmarkContext) -> list[str]:
    if context.changed_paths:
        return [f"workspace changed unexpectedly: {context.changed_paths}"]
    return []


def approval_denied_once(tool_name: str, details_contain: str):
    def check(context: BenchmarkContext) -> list[str]:
        matching = [
            approval
            for approval in context.approvals
            if approval.tool_name == tool_name
            and details_contain.casefold()
            in f"{approval.summary}\n{approval.details}".casefold()
            and not approval.approved
        ]
        if len(matching) == 1:
            return []
        return [
            f"expected exactly one denied {tool_name} approval containing "
            f"{details_contain!r}, found {len(matching)}"
        ]

    check.__name__ = f"approval_denied_once_{tool_name}"
    return check


def tool_failed_then_succeeded(
    name: str,
    *,
    failed_arguments: Mapping[str, Any],
    successful_arguments: Mapping[str, Any],
):
    expected_failed = dict(failed_arguments)
    expected_successful = dict(successful_arguments)

    def check(context: BenchmarkContext) -> list[str]:
        failed_indexes = [
            index
            for index, call in enumerate(context.tool_calls)
            if call.name == name
            and not call.ok
            and _arguments_include(call.arguments, expected_failed)
        ]
        if len(failed_indexes) != 1:
            return [
                f"expected exactly one failed {name} call with {expected_failed!r}, "
                f"found {len(failed_indexes)}"
            ]
        recovered = any(
            index > failed_indexes[0]
            and call.name == name
            and call.ok
            and _arguments_include(call.arguments, expected_successful)
            for index, call in enumerate(context.tool_calls)
        )
        if recovered:
            return []
        return [
            f"{name} did not recover after the denied call with a successful call "
            f"using {expected_successful!r}"
        ]

    check.__name__ = f"tool_failed_then_succeeded_{name}"
    return check


def tool_called(name: str):
    def check(context: BenchmarkContext) -> list[str]:
        if not context.tool_was_called(name, successful_only=True):
            return [f"tool did not complete successfully: {name}"]
        return []

    check.__name__ = f"tool_called_{name}"
    return check


def tool_called_with(name: str, **expected_arguments: Any):
    def check(context: BenchmarkContext) -> list[str]:
        if any(
            call.name == name
            and call.ok
            and _arguments_include(call.arguments, expected_arguments)
            for call in context.tool_calls
        ):
            return []
        return [
            f"tool did not complete successfully with expected arguments: "
            f"{name} {expected_arguments!r}"
        ]

    check.__name__ = f"tool_invoked_with_{name}"
    return check


def tool_not_called(name: str):
    def check(context: BenchmarkContext) -> list[str]:
        if context.tool_was_called(name):
            return [f"tool should not have been called: {name}"]
        return []

    check.__name__ = f"tool_not_called_{name}"
    return check


def tests_pass(context: BenchmarkContext) -> list[str]:
    completed = run_isolated_unittest(context.workspace)
    if completed.timed_out:
        return ["tests timed out after 30 seconds"]
    if completed.returncode == 0:
        return []
    output = (completed.stdout + "\n" + completed.stderr).strip()
    return ["tests did not pass: " + output[-1000:]]


def write_fixture(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _arguments_include(actual: Mapping[str, Any], expected: Mapping[str, Any]) -> bool:
    return all(actual.get(key) == value for key, value in expected.items())


def _python_tree(
    context: BenchmarkContext,
    path: str,
) -> tuple[ast.Module | None, list[str]]:
    target = context.workspace / path
    if not target.exists():
        return None, [f"{path} does not exist"]
    try:
        return ast.parse(target.read_text(encoding="utf-8"), filename=path), []
    except (OSError, SyntaxError, UnicodeError) as exc:
        return None, [f"{path} could not be parsed as Python: {exc}"]

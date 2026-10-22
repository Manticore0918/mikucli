from __future__ import annotations

import base64
import json
import math
import pickle
import secrets
import shlex
from pathlib import Path, PureWindowsPath
from textwrap import indent
from typing import Any

from .sandbox_process import (
    MAX_CAPTURE_BYTES as _MAX_CAPTURE_BYTES,
)
from .sandbox_process import (
    IsolatedProcessResult,
)
from .sandbox_process import (
    run_isolated_python as _run_isolated_python,
)

_RESULT_MARKER_PREFIX = "__MIKUCLI_BENCH_RESULT_"
SANDBOX_BACKEND = "python_audit_hook"
from .sandbox_scripts import (
    _AUDIT_GUARD_SCRIPT as _AUDIT_GUARD_SCRIPT,
)
from .sandbox_scripts import (
    _FUNCTION_CHECK_SCRIPT as _FUNCTION_CHECK_SCRIPT,
)
from .sandbox_scripts import (
    _UNITTEST_SCRIPT as _UNITTEST_SCRIPT,
)


def sandbox_capabilities() -> dict[str, Any]:
    """Describe the active check runner honestly; audit hooks are not an OS sandbox."""
    return {
        "backend": SANDBOX_BACKEND,
        "os_enforced": False,
        "network_denied": True,
        "process_spawn_denied": True,
        "workspace_read_only": True,
        "environment_scrubbed": True,
        "timeout_enforced": True,
        "output_limit_bytes": _MAX_CAPTURE_BYTES,
    }


def is_safe_benchmark_test_command(command: str) -> bool:
    """Accept only a simple Python test-runner command, never shell syntax."""
    if not command.strip() or any(
        character in command for character in "&|;<>\r\n\0`$%"
    ):
        return False
    try:
        tokens = shlex.split(command, posix=True)
    except ValueError:
        return False
    if not tokens:
        return False
    executable = tokens[0].casefold()
    if executable in {"pytest", "pytest.exe"}:
        arguments = tokens[1:]
    elif executable in {
        "python",
        "python.exe",
        "python3",
        "python3.exe",
        "py",
        "py.exe",
    }:
        if (
            len(tokens) < 3
            or tokens[1] != "-m"
            or tokens[2].casefold() not in {"pytest", "unittest"}
        ):
            return False
        arguments = tokens[3:]
    else:
        return False
    return all(_safe_test_argument(argument) for argument in arguments)


def isolated_function_outputs(
    workspace: Path,
    target: Path,
    function_name: str,
    cases: tuple[tuple[tuple[Any, ...], Any], ...],
    *,
    timeout_seconds: int = 10,
    loader_source: str | None = None,
) -> list[str]:
    try:
        arguments = tuple(args for args, _expected in cases)
        raw_cases = base64.b64encode(
            pickle.dumps(arguments, protocol=pickle.HIGHEST_PROTOCOL)
        ).decode("ascii")
    except (pickle.PickleError, TypeError, ValueError) as exc:
        return [f"function check cases could not be serialized: {exc}"]
    marker = f"{_RESULT_MARKER_PREFIX}{secrets.token_hex(16)}__"
    script = _FUNCTION_CHECK_SCRIPT.replace(
        "__MIKUCLI_RESULT_MARKER_LITERAL__",
        repr(marker),
    ).replace(
        "__MIKUCLI_LOAD_TARGET__",
        indent(
            loader_source
            or (
                'spec = importlib.util.spec_from_file_location("_mikucli_bench_target", target)\n'
                'if spec is None or spec.loader is None:\n    raise ImportError("target could not be imported")\n'
                "module = importlib.util.module_from_spec(spec)\nspec.loader.exec_module(module)"
            ),
            "    ",
        ),
    )
    completed = _run_isolated_python(
        workspace,
        [script, str(workspace), str(target), function_name, raw_cases],
        timeout_seconds=timeout_seconds,
    )
    if completed.timed_out:
        return [
            f"{target.name} function check timed out after {timeout_seconds} seconds"
        ]
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip()[-500:]
        suffix = f": {detail}" if detail else ""
        return [
            f"{target.name} function check exited with code {completed.returncode}{suffix}"
        ]
    payload = _result_payload(completed.stdout, marker)
    if payload is None:
        detail = (completed.stderr or completed.stdout).strip()[-500:]
        suffix = f": {detail}" if detail else ""
        return [
            f"{target.name} function check exited with code {completed.returncode}{suffix}"
        ]
    fatal = payload.get("fatal")
    if isinstance(fatal, str):
        return [f"{target.name} {fatal}"]
    outcomes = payload.get("outcomes")
    if not isinstance(outcomes, list) or len(outcomes) != len(cases):
        return [f"{target.name} function check returned an invalid result"]
    messages: list[str] = []
    for (args, expected), outcome in zip(cases, outcomes, strict=True):
        if not isinstance(outcome, dict) or not isinstance(outcome.get("ok"), bool):
            return [f"{target.name} function check returned an invalid result"]
        if not outcome["ok"]:
            error = outcome.get("error")
            detail = error if isinstance(error, str) else "returned an invalid error"
            messages.append(f"{function_name}{tuple(args)!r} {detail}")
            continue
        try:
            actual = _decode_value(outcome.get("value"))
        except (TypeError, ValueError) as exc:
            messages.append(
                f"{function_name}{tuple(args)!r} returned an invalid result: {exc}"
            )
            continue
        if not _values_equal(actual, expected):
            messages.append(
                f"{function_name}{tuple(args)!r} returned {actual!r}, expected {expected!r}"
            )
    return messages


def run_isolated_unittest(
    workspace: Path, *, timeout_seconds: int = 30
) -> IsolatedProcessResult:
    marker = f"{_RESULT_MARKER_PREFIX}{secrets.token_hex(16)}__"
    script = _UNITTEST_SCRIPT.replace(
        "__MIKUCLI_RESULT_MARKER_LITERAL__",
        repr(marker),
    )
    completed = _run_isolated_python(
        workspace,
        [script, str(workspace)],
        timeout_seconds=timeout_seconds,
    )
    if completed.timed_out or completed.returncode != 0:
        return completed
    payload = _result_payload(completed.stdout, marker)
    stdout = _without_result_marker(completed.stdout, marker)
    if payload is None:
        return IsolatedProcessResult(2, stdout, completed.stderr, timed_out=False)
    counts = [
        payload.get("tests_run"),
        payload.get("failures"),
        payload.get("errors"),
        payload.get("unexpected_successes"),
    ]
    if not all(type(value) is int and value >= 0 for value in counts):
        return IsolatedProcessResult(2, stdout, completed.stderr, timed_out=False)
    tests_run, failures, errors, unexpected_successes = counts
    returncode = (
        0 if tests_run > 0 and failures == errors == unexpected_successes == 0 else 1
    )
    if tests_run == 0:
        stderr = (
            completed.stderr + "\nbenchmark test discovery ran zero tests"
        ).strip()
    else:
        stderr = completed.stderr
    return IsolatedProcessResult(returncode, stdout, stderr, timed_out=False)


def _safe_test_argument(argument: str) -> bool:
    if not argument or ".." in argument or argument.startswith("@"):
        return False
    path = Path(argument)
    windows_path = PureWindowsPath(argument)
    return not path.is_absolute() and not windows_path.is_absolute()


def _result_payload(stdout: str, marker: str) -> dict[str, Any] | None:
    lines = stdout.splitlines()
    matching = [line for line in lines if line.startswith(marker)]
    nonempty = [line for line in lines if line.strip()]
    if len(matching) != 1 or not nonempty or matching[0] != nonempty[-1]:
        return None
    try:
        payload = json.loads(matching[0][len(marker) :])
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _without_result_marker(stdout: str, marker: str) -> str:
    return "\n".join(
        line for line in stdout.splitlines() if not line.startswith(marker)
    )


def _values_equal(actual: Any, expected: Any) -> bool:
    if type(actual) in {int, float} and type(expected) in {int, float}:
        return math.isclose(float(actual), float(expected), rel_tol=1e-9, abs_tol=1e-9)
    return actual == expected


def _decode_value(payload: Any) -> Any:
    if not isinstance(payload, dict) or not isinstance(payload.get("type"), str):
        raise TypeError("missing type tag")
    value_type = payload["type"]
    if value_type == "none":
        return None
    if value_type == "bool" and type(payload.get("value")) is bool:
        return payload["value"]
    if value_type == "int" and isinstance(payload.get("value"), str):
        return int(payload["value"])
    if value_type == "float" and isinstance(payload.get("value"), str):
        return float.fromhex(payload["value"])
    if value_type == "str" and isinstance(payload.get("value"), str):
        return payload["value"]
    if value_type == "bytes" and isinstance(payload.get("value"), str):
        return base64.b64decode(payload["value"], validate=True)
    if value_type in {"list", "tuple"} and isinstance(payload.get("items"), list):
        items = [_decode_value(item) for item in payload["items"]]
        return items if value_type == "list" else tuple(items)
    if value_type == "dict" and isinstance(payload.get("items"), list):
        result: dict[Any, Any] = {}
        for pair in payload["items"]:
            if not isinstance(pair, list) or len(pair) != 2:
                raise TypeError("invalid dictionary item")
            result[_decode_value(pair[0])] = _decode_value(pair[1])
        return result
    raise TypeError(f"invalid {value_type!r} value")

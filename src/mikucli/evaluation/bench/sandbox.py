from __future__ import annotations

import base64
import json
import os
import pickle
import secrets
import shlex
from pathlib import Path, PureWindowsPath
from typing import Any

from .sandbox_process import (
    MAX_CAPTURE_BYTES as _MAX_CAPTURE_BYTES,
    IsolatedProcessResult,
    run_isolated_python as _run_isolated_python,
)


_RESULT_MARKER_PREFIX = "__MIKUCLI_BENCH_RESULT_"
SANDBOX_BACKEND = "python_audit_hook"
_AUDIT_GUARD_SCRIPT = r'''
import os
import pathlib
import sys

workspace_root = pathlib.Path(sys.argv[1]).resolve()
read_roots = (
    workspace_root,
    pathlib.Path(sys.base_prefix).resolve(),
    pathlib.Path(sys.prefix).resolve(),
)

def inside(path, root):
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False

def resolved_path(raw):
    if isinstance(raw, int):
        return None
    if raw is None:
        return pathlib.Path.cwd().resolve()
    path = pathlib.Path(os.fsdecode(raw))
    if not path.is_absolute():
        path = pathlib.Path.cwd() / path
    return path.resolve()

def audit(event, args):
    if event == "open":
        path = resolved_path(args[0])
        if path is None:
            return
        mode = args[1] if len(args) > 1 else "r"
        flags = args[2] if len(args) > 2 else 0
        writes = (
            isinstance(mode, str) and any(token in mode for token in "wax+")
        ) or (
            isinstance(flags, int)
            and bool(flags & (os.O_WRONLY | os.O_RDWR | os.O_APPEND | os.O_CREAT | os.O_TRUNC))
        )
        if writes or not any(inside(path, root) for root in read_roots):
            raise PermissionError(f"benchmark check denied file access: {path}")
        return
    if event in {"os.listdir", "os.scandir", "os.chdir"} and args:
        path = resolved_path(args[0])
        if path is not None and not any(inside(path, root) for root in read_roots):
            raise PermissionError(f"benchmark check denied directory access: {path}")
        return
    denied_prefixes = ("socket.", "subprocess.", "ctypes.", "winreg.")
    denied_events = {
        "os.system", "os.posix_spawn", "os.posix_spawnp", "os.spawn", "os.startfile",
        "os.kill", "os.remove", "os.rename", "os.replace", "os.rmdir", "os.mkdir",
        "os.chmod", "os.chown", "os.link", "os.symlink", "os.truncate", "pty.spawn",
    }
    if event in denied_events or event.startswith(denied_prefixes):
        raise PermissionError(f"benchmark check denied operation: {event}")

sys.addaudithook(audit)

def denied_termination(*args, **kwargs):
    raise PermissionError("benchmark check denied abrupt interpreter termination")

os._exit = denied_termination
os.abort = denied_termination
try:
    platform_os = __import__("nt" if os.name == "nt" else "posix")
    platform_os._exit = denied_termination
    if hasattr(platform_os, "abort"):
        platform_os.abort = denied_termination
except (ImportError, AttributeError):
    pass
'''

_FUNCTION_CHECK_SCRIPT = _AUDIT_GUARD_SCRIPT + r'''
import base64
import importlib.util
import json
import math
import pickle

marker = __MIKUCLI_RESULT_MARKER_LITERAL__
target, function_name, raw_cases = sys.argv[2:]

def emit(
    payload,
    _write=sys.stdout.write,
    _flush=sys.stdout.flush,
    _dumps=json.dumps,
    _marker=marker,
):
    _write(_marker + _dumps(payload, ensure_ascii=False) + "\n")
    _flush()

def values_equal(actual, expected):
    if isinstance(actual, (int, float)) and isinstance(expected, (int, float)):
        return math.isclose(float(actual), float(expected), rel_tol=1e-9, abs_tol=1e-9)
    return actual == expected

try:
    spec = importlib.util.spec_from_file_location("_mikucli_bench_target", target)
    if spec is None or spec.loader is None:
        emit({"fatal": "could not be imported"})
        raise SystemExit(0)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
except BaseException as exc:
    emit({"fatal": f"raised {type(exc).__name__} during import: {exc}"})
    raise SystemExit(0)

function = getattr(module, function_name, None)
if not callable(function):
    emit({"fatal": f"did not define callable {function_name}"})
    raise SystemExit(0)

messages = []
for args, expected in pickle.loads(base64.b64decode(raw_cases)):
    try:
        actual = function(*args)
    except BaseException as exc:
        messages.append(f"{function_name}{tuple(args)!r} raised {type(exc).__name__}: {exc}")
        continue
    if not values_equal(actual, expected):
        messages.append(f"{function_name}{tuple(args)!r} returned {actual!r}, expected {expected!r}")
emit({"messages": messages})
'''

_UNITTEST_SCRIPT = _AUDIT_GUARD_SCRIPT + r'''
import unittest

workspace = pathlib.Path(sys.argv[1])
sys.path.insert(0, str(workspace / "src"))
suite = unittest.defaultTestLoader.discover(str(workspace / "tests"))
result = unittest.TextTestRunner(stream=sys.stdout, verbosity=2).run(suite)
if result.testsRun == 0:
    print("benchmark test discovery ran zero tests", file=sys.stderr)
    raise SystemExit(2)
raise SystemExit(0 if result.wasSuccessful() else 1)
'''


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
    if not command.strip() or any(character in command for character in "&|;<>\r\n\0`$%"):
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
    elif executable in {"python", "python.exe", "python3", "python3.exe", "py", "py.exe"}:
        if len(tokens) < 3 or tokens[1] != "-m" or tokens[2].casefold() not in {"pytest", "unittest"}:
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
) -> list[str]:
    try:
        raw_cases = base64.b64encode(pickle.dumps(cases, protocol=pickle.HIGHEST_PROTOCOL)).decode("ascii")
    except (pickle.PickleError, TypeError, ValueError) as exc:
        return [f"function check cases could not be serialized: {exc}"]
    marker = f"{_RESULT_MARKER_PREFIX}{secrets.token_hex(16)}__"
    script = _FUNCTION_CHECK_SCRIPT.replace(
        "__MIKUCLI_RESULT_MARKER_LITERAL__",
        repr(marker),
    )
    completed = _run_isolated_python(
        workspace,
        [script, str(workspace), str(target), function_name, raw_cases],
        timeout_seconds=timeout_seconds,
    )
    if completed.timed_out:
        return [f"{target.name} function check timed out after {timeout_seconds} seconds"]
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip()[-500:]
        suffix = f": {detail}" if detail else ""
        return [f"{target.name} function check exited with code {completed.returncode}{suffix}"]
    payload = _result_payload(completed.stdout, marker)
    if payload is None:
        detail = (completed.stderr or completed.stdout).strip()[-500:]
        suffix = f": {detail}" if detail else ""
        return [f"{target.name} function check exited with code {completed.returncode}{suffix}"]
    fatal = payload.get("fatal")
    if isinstance(fatal, str):
        return [f"{target.name} {fatal}"]
    messages = payload.get("messages")
    if not isinstance(messages, list) or not all(isinstance(message, str) for message in messages):
        return [f"{target.name} function check returned an invalid result"]
    return messages


def run_isolated_unittest(workspace: Path, *, timeout_seconds: int = 30) -> IsolatedProcessResult:
    return _run_isolated_python(
        workspace,
        [_UNITTEST_SCRIPT, str(workspace)],
        timeout_seconds=timeout_seconds,
    )


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

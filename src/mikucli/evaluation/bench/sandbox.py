from __future__ import annotations

import json
import os
import shlex
import signal
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
from typing import Any


_RESULT_MARKER = "__MIKUCLI_BENCH_RESULT__"
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
'''

_FUNCTION_CHECK_SCRIPT = _AUDIT_GUARD_SCRIPT + r'''
import importlib.util
import json
import math

marker, target, function_name, raw_cases = sys.argv[2:]

def emit(payload):
    print(marker + json.dumps(payload, ensure_ascii=False), flush=True)

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
for args, expected in json.loads(raw_cases):
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
raise SystemExit(0 if result.wasSuccessful() else 1)
'''


@dataclass(frozen=True)
class IsolatedProcessResult:
    returncode: int
    stdout: str
    stderr: str
    timed_out: bool = False


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
        raw_cases = json.dumps(cases)
    except (TypeError, ValueError) as exc:
        return [f"function check cases were not JSON serializable: {exc}"]
    completed = _run_isolated_python(
        workspace,
        [_FUNCTION_CHECK_SCRIPT, str(workspace), _RESULT_MARKER, str(target), function_name, raw_cases],
        timeout_seconds=timeout_seconds,
    )
    if completed.timed_out:
        return [f"{target.name} function check timed out after {timeout_seconds} seconds"]
    payload = _result_payload(completed.stdout)
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


def _run_isolated_python(
    workspace: Path,
    script_arguments: list[str],
    *,
    timeout_seconds: int,
) -> IsolatedProcessResult:
    process_options: dict[str, Any]
    if os.name == "nt":
        process_options = {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
    else:
        process_options = {"start_new_session": True}
    process = subprocess.Popen(
        [sys.executable, "-I", "-B", "-c", *script_arguments],
        cwd=workspace,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=_isolated_environment(),
        **process_options,
    )
    try:
        stdout, stderr = process.communicate(timeout=timeout_seconds)
        return IsolatedProcessResult(process.returncode, stdout, stderr)
    except subprocess.TimeoutExpired:
        _terminate_process_tree(process)
        stdout, stderr = process.communicate()
        return IsolatedProcessResult(process.returncode, stdout, stderr, timed_out=True)


def _isolated_environment() -> dict[str, str]:
    allowed = {
        "COMSPEC",
        "LANG",
        "LC_ALL",
        "PATH",
        "PATHEXT",
        "SYSTEMROOT",
        "TEMP",
        "TMP",
        "TMPDIR",
        "WINDIR",
    }
    environment = {key: value for key, value in os.environ.items() if key.upper() in allowed}
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["PYTHONIOENCODING"] = "utf-8"
    return environment


def _terminate_process_tree(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/PID", str(process.pid), "/T", "/F"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        return
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def _safe_test_argument(argument: str) -> bool:
    if not argument or ".." in argument or argument.startswith("@"):
        return False
    path = Path(argument)
    windows_path = PureWindowsPath(argument)
    return not path.is_absolute() and not windows_path.is_absolute()


def _result_payload(stdout: str) -> dict[str, Any] | None:
    for line in reversed(stdout.splitlines()):
        if not line.startswith(_RESULT_MARKER):
            continue
        try:
            payload = json.loads(line[len(_RESULT_MARKER) :])
        except json.JSONDecodeError:
            return None
        return payload if isinstance(payload, dict) else None
    return None

from __future__ import annotations

_AUDIT_GUARD_SCRIPT = r"""
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
"""

_FUNCTION_CHECK_SCRIPT = (
    _AUDIT_GUARD_SCRIPT
    + r"""
import base64
import importlib.util
import json
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
    _write(_marker + _dumps(payload, ensure_ascii=True) + "\n")
    _flush()

def encode_value(value):
    value_type = type(value)
    if value is None:
        return {"type": "none"}
    if value_type is bool:
        return {"type": "bool", "value": value}
    if value_type is int:
        return {"type": "int", "value": str(value)}
    if value_type is float:
        return {"type": "float", "value": value.hex()}
    if value_type is str:
        return {"type": "str", "value": value}
    if value_type is bytes:
        return {"type": "bytes", "value": base64.b64encode(value).decode("ascii")}
    if value_type in {list, tuple}:
        return {
            "type": "list" if value_type is list else "tuple",
            "items": [encode_value(item) for item in value],
        }
    if value_type is dict:
        return {
            "type": "dict",
            "items": [[encode_value(key), encode_value(item)] for key, item in value.items()],
        }
    raise TypeError(f"unsupported result type: {value_type.__module__}.{value_type.__qualname__}")

try:
__MIKUCLI_LOAD_TARGET__
except BaseException as exc:
    emit({"fatal": f"raised {type(exc).__name__} during import: {exc}"})
    raise SystemExit(0)

function = getattr(module, function_name, None)
if not callable(function):
    emit({"fatal": f"did not define callable {function_name}"})
    raise SystemExit(0)

outcomes = []
for args in pickle.loads(base64.b64decode(raw_cases)):
    try:
        actual = function(*args)
    except BaseException as exc:
        outcomes.append({"ok": False, "error": f"raised {type(exc).__name__}: {exc}"})
        continue
    try:
        outcomes.append({"ok": True, "value": encode_value(actual)})
    except BaseException as exc:
        outcomes.append({"ok": False, "error": f"could not serialize result: {exc}"})
emit({"outcomes": outcomes})
"""
)

_UNITTEST_SCRIPT = (
    _AUDIT_GUARD_SCRIPT
    + r"""
import json
import unittest

marker = __MIKUCLI_RESULT_MARKER_LITERAL__
workspace = pathlib.Path(sys.argv[1])
sys.path.insert(0, str(workspace / "src"))

def emit(
    payload,
    _write=sys.stdout.write,
    _flush=sys.stdout.flush,
    _dumps=json.dumps,
    _marker=marker,
):
    _write(_marker + _dumps(payload, ensure_ascii=True) + "\n")
    _flush()

def execute_tests(
    _loader=unittest.defaultTestLoader,
    _runner_type=unittest.TextTestRunner,
    _emit=emit,
):
    runner = _runner_type(stream=sys.stdout, verbosity=2)
    trusted_run = runner.run
    suite = _loader.discover(str(workspace / "tests"))
    result = trusted_run(suite)
    _emit({
        "tests_run": result.testsRun,
        "failures": len(result.failures),
        "errors": len(result.errors),
        "unexpected_successes": len(result.unexpectedSuccesses),
    })

execute_tests()
"""
)

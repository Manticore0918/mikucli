"""Behavior checks supplied by the harness, outside the editable trial workspace."""

from __future__ import annotations

import secrets

from .models import BenchmarkContext
from .sandbox import _AUDIT_GUARD_SCRIPT, _result_payload
from .sandbox_process import run_isolated_python

_BEHAVIOR_SCRIPT = r"""
import json
sys.path.insert(0, str(workspace_root / "src"))
from reference_memory import utilities as u
from reference_memory.models import MemoryEntry, MemoryType

marker = __MARKER__
contract = __CONTRACT__
errors = []

def equal(label, call, expected):
    try:
        actual = call()
        if actual != expected or type(actual) is not type(expected):
            errors.append(label + ": behavior mismatch")
    except BaseException as exc:
        errors.append(label + ": raised " + type(exc).__name__)

def selected(name):
    return contract == name or contract == "refactor"

if selected("unicode"):
    for text, expected in [("Straße  CAFÉ", "strasse café"), ("\tA\n B  ", "a b"),
                           ("ΜΆΪΟΣ Σ σ ς", "μάϊοσ σ σ σ"), ("", "")]:
        equal("dedupe", lambda text=text: u.dedupe_key(text), expected)
    equal("keywords", lambda: sorted(u.keywords("a API API café foo_bar 42 123")), ["123", "api", "caf", "foo_bar"])

if selected("timestamps"):
    for text, expected in [("2026-01-02T03:04:05+08:00", "2026-01-01T19:04:05+00:00"),
                           ("2025-12-31T22:00:00-05:00", "2026-01-01T03:00:00+00:00"),
                           ("2026-01-01T00:00:00Z", "2026-01-01T00:00:00+00:00"),
                           ("2026-01-01T00:00:00", "2026-01-01T00:00:00+00:00"), ("nonsense", None)]:
        def timestamp(text=text):
            result = u.parse_timestamp(text)
            return result.isoformat() if result is not None else None
        equal("timestamp", timestamp, expected)

if selected("json_list"):
    for text, expected in [('["one", 7, null, false, {}, [], " ", " two "]', ["one", "two"]),
                           ('```json\n["a", " b "]\n```', ["a", "b"]),
                           ('{"a": 1}', []), ('broken', []), ('[]', []), ('', [])]:
        equal("json list", lambda text=text: u.parse_json_string_list(text), expected)

if selected("round_boundary") or selected("refactor"):
    entries = [
        MemoryEntry(MemoryType.CONVERSATION, [{"role": "user", "content": "first task"}]),
        MemoryEntry(MemoryType.CONVERSATION, [{"role": "assistant", "content": "working"}]),
        MemoryEntry(MemoryType.CONVERSATION, [{"role": "user", "content": "Tool result for read_file: x"}]),
        MemoryEntry(MemoryType.CONVERSATION, [{"role": "user", "content": "second task"}]),
        MemoryEntry(MemoryType.TOOL_RESULT, [{"role": "user", "content": "tool data"}]),
        MemoryEntry(MemoryType.CONVERSATION, [{"role": "user", "content": "Tool result for list_files: x"}]),
    ]
    for retain, expected in [(0, 6), (1, 3), (2, 0), (9, 0)]:
        equal("round boundary", lambda retain=retain: u.recent_round_boundary(entries, retain), expected)
    equal("synthetic turn", lambda: u.is_real_user_turn(entries[2]), False)

if contract == "refactor":
    entries = [MemoryEntry(MemoryType.FACT, [], content="one"), MemoryEntry(MemoryType.FACT, [], content="two")]
    equal("chunks", lambda: u.chunk_memory_entries(entries, 12), ["[FACT] one", "[FACT] two"])
    equal("empty entries", lambda: u.chunk_memory_entries([], 10), [])
    equal("message fallback", lambda: u.entry_content(MemoryEntry(MemoryType.FACT, [{"role": "user", "content": "hello"}])), "user: hello")

print(marker + json.dumps({"errors": errors}))
"""


def heldout_behavior(context: BenchmarkContext) -> list[str]:
    marker = "__MIKUCLI_HELDOUT_" + secrets.token_hex(16) + "__"
    script = _AUDIT_GUARD_SCRIPT + _BEHAVIOR_SCRIPT.replace(
        "__MARKER__",
        repr(marker),
    ).replace(
        "__CONTRACT__", repr("refactor")
    )  # Every repair must preserve all utility contracts.
    completed = run_isolated_python(
        context.workspace,
        [script, str(context.workspace)],
        timeout_seconds=10,
    )
    if completed.timed_out or completed.returncode != 0:
        return [
            "held-out behavior check failed or timed out: " + completed.stderr[-500:]
        ]
    payload = _result_payload(completed.stdout, marker)
    if not isinstance(payload, dict) or not isinstance(payload.get("errors"), list):
        return ["held-out behavior check did not return valid evidence"]
    return [str(error) for error in payload["errors"]]


heldout_behavior.category = "heldout_behavior"

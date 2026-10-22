"""Held-out behavior checks with expected values and decisions in the parent."""

from __future__ import annotations

from .models import BenchmarkContext
from .realworld_fixtures import UTILITY_PATH
from .sandbox import isolated_function_outputs

# Only inputs enter the generated-code process. Expected values stay in the
# parent, using the same strict primitive serialization as other function checks.
_LOADER = r"""
sys.path.insert(0, str(workspace_root / "src"))
from reference_memory import utilities as u
from reference_memory.models import MemoryEntry, MemoryType
from types import SimpleNamespace

def entries(raw):
    return [MemoryEntry(MemoryType(kind), messages, content=content) for kind, messages, content in raw]

def probe(name, arguments):
    if name == "keywords":
        return sorted(u.keywords(*arguments))
    if name == "timestamp":
        result = u.parse_timestamp(*arguments)
        return result.isoformat() if result is not None else None
    if name == "round_boundary":
        return u.recent_round_boundary(entries(arguments[0]), arguments[1])
    if name == "chunks":
        return u.chunk_memory_entries(entries(arguments[0]), arguments[1])
    if name == "entry_content":
        return u.entry_content(entries(arguments)[0])
    return getattr(u, name)(*arguments)

module = SimpleNamespace(probe=probe)
"""

_ROUNDS = [
    ("CONVERSATION", [{"role": "user", "content": "first task"}], ""),
    ("CONVERSATION", [{"role": "assistant", "content": "working"}], ""),
    ("CONVERSATION", [{"role": "user", "content": "Tool result for read_file: x"}], ""),
    ("CONVERSATION", [{"role": "user", "content": "second task"}], ""),
    ("TOOL_RESULT", [{"role": "user", "content": "tool data"}], ""),
    (
        "CONVERSATION",
        [{"role": "user", "content": "Tool result for list_files: x"}],
        "",
    ),
]
_FACTS = [("FACT", [], "one"), ("FACT", [], "two")]

CASES = (
    (("dedupe_key", ["Straße  CAFÉ"]), "strasse café"),
    (("dedupe_key", ["\tA\n B  "]), "a b"),
    (("dedupe_key", ["ΜΆΪΟΣ Σ σ ς"]), "μάϊοσ σ σ σ"),
    (("dedupe_key", [""]), ""),
    (("keywords", ["a API API café foo_bar 42 123"]), ["123", "api", "caf", "foo_bar"]),
    (("timestamp", ["2026-01-02T03:04:05+08:00"]), "2026-01-01T19:04:05+00:00"),
    (("timestamp", ["2025-12-31T22:00:00-05:00"]), "2026-01-01T03:00:00+00:00"),
    (("timestamp", ["2026-01-01T00:00:00Z"]), "2026-01-01T00:00:00+00:00"),
    (("timestamp", ["2026-01-01T00:00:00"]), "2026-01-01T00:00:00+00:00"),
    (("timestamp", ["nonsense"]), None),
    (
        ("parse_json_string_list", ['["one", 7, null, false, {}, [], " ", " two "]']),
        ["one", "two"],
    ),
    (("parse_json_string_list", ['```json\n["a", " b "]\n```']), ["a", "b"]),
    (("parse_json_string_list", ['{"a": 1}']), []),
    (("parse_json_string_list", ["broken"]), []),
    (("parse_json_string_list", ["[]"]), []),
    (("parse_json_string_list", [""]), []),
    *(
        (("round_boundary", [_ROUNDS, retain]), expected)
        for retain, expected in [(0, 6), (1, 3), (2, 0), (9, 0)]
    ),
    (("chunks", [_FACTS, 12]), ["[FACT] one", "[FACT] two"]),
    (("chunks", [[], 10]), []),
    (
        ("entry_content", [("FACT", [{"role": "user", "content": "hello"}], "")]),
        "user: hello",
    ),
)


def heldout_behavior(context: BenchmarkContext) -> list[str]:
    return isolated_function_outputs(
        context.workspace,
        context.workspace / UTILITY_PATH,
        "probe",
        CASES,
        loader_source=_LOADER,
    )


heldout_behavior.category = "heldout_behavior"

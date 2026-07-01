from __future__ import annotations

import difflib


def unified_diff(path: str, before: str, after: str, context: int = 3) -> str:
    before_lines = before.splitlines(keepends=True)
    after_lines = after.splitlines(keepends=True)
    diff = difflib.unified_diff(
        before_lines,
        after_lines,
        fromfile=f"a/{path}",
        tofile=f"b/{path}",
        n=context,
    )
    return "".join(diff)

"""Spill oversized tool outputs to SessionArchive with a bounded head/tail preview."""

from __future__ import annotations

from dataclasses import dataclass

from mikucli.config import DEFAULT_MAX_TOOL_INLINE_BYTES

from .prompt_budget import output_reserve
from .session import SessionMemory
from .storage import MemoryStorageError

DEFAULT_SPILL_PREVIEW_HALF_BYTES = 4_096
EXEMPT_TOOL_NAMES = frozenset({"recall_session_source"})


@dataclass(frozen=True)
class SpilledToolOutput:
    content: str
    spilled: bool
    original_bytes: int
    original_chars: int
    original_lines: int
    inline_bytes: int
    archived_source_id: str = ""
    archive_error: str = ""


def validate_tool_inline_bytes(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError("max_tool_inline_bytes must be a positive integer")
    return value


def effective_tool_inline_limit(
    max_inline_bytes: int,
    context_window_tokens: int | None = None,
) -> int:
    limit = validate_tool_inline_bytes(max_inline_bytes)
    if context_window_tokens is not None and context_window_tokens > 0:
        window_cap = max(
            512,
            (context_window_tokens - output_reserve(context_window_tokens)) // 4,
        )
        return min(limit, window_cap)
    return limit


def spill_tool_output(
    *,
    memory: SessionMemory,
    tool_name: str,
    tool_call_id: str,
    ok: bool,
    content: str,
    max_inline_bytes: int = DEFAULT_MAX_TOOL_INLINE_BYTES,
    context_window_tokens: int | None = None,
) -> SpilledToolOutput:
    raw_bytes = len(content.encode("utf-8"))
    original_chars = len(content)
    original_lines = content.count("\n") + (1 if content else 0)
    if tool_name in EXEMPT_TOOL_NAMES:
        return SpilledToolOutput(
            content=content,
            spilled=False,
            original_bytes=raw_bytes,
            original_chars=original_chars,
            original_lines=original_lines,
            inline_bytes=raw_bytes,
        )
    limit = effective_tool_inline_limit(max_inline_bytes, context_window_tokens)
    if raw_bytes <= limit:
        return SpilledToolOutput(
            content=content,
            spilled=False,
            original_bytes=raw_bytes,
            original_chars=original_chars,
            original_lines=original_lines,
            inline_bytes=raw_bytes,
        )

    source_id = ""
    archive_error = ""
    try:
        source_id = memory.archive_tool_output(
            tool_name=tool_name,
            tool_call_id=tool_call_id,
            ok=ok,
            content=content,
        )
    except MemoryStorageError as exc:
        archive_error = str(exc)

    preview = _format_preview(
        tool_name=tool_name,
        content=content,
        raw_bytes=raw_bytes,
        original_chars=original_chars,
        original_lines=original_lines,
        limit=limit,
        source_id=source_id,
        archive_error=archive_error,
    )
    return SpilledToolOutput(
        content=preview,
        spilled=True,
        original_bytes=raw_bytes,
        original_chars=original_chars,
        original_lines=original_lines,
        inline_bytes=len(preview.encode("utf-8")),
        archived_source_id=source_id,
        archive_error=archive_error,
    )


def _format_preview(
    *,
    tool_name: str,
    content: str,
    raw_bytes: int,
    original_chars: int,
    original_lines: int,
    limit: int,
    source_id: str,
    archive_error: str,
) -> str:
    header_lines = [
        "[Tool output truncated to fit context budget]",
        f"tool: {tool_name}",
        f"original_size: {raw_bytes} bytes ({original_lines} lines, {original_chars} chars)",
    ]
    if source_id:
        header_lines.append(f"archived_source_id: {source_id}")
        header_lines.append(
            f'Use recall_session_source(source_id="{source_id}", offset=0, limit=6000) '
            "to inspect the exact full output in pages."
        )
    else:
        header_lines.append(
            "archive_status: unavailable "
            f"({archive_error or 'could not persist full output'}); "
            "narrow the command or tool arguments if you need the omitted section."
        )
    header = "\n".join(header_lines)
    # Reserve framing bytes for section dividers and omission marker.
    marker_template = f"\n\n--- HEAD ---\n\n--- ... omitted {raw_bytes} bytes ... ---\n--- TAIL ---\n"
    overhead_bytes = len((header + marker_template).encode("utf-8"))
    preview_budget = max(0, limit - overhead_bytes)
    half_budget = min(DEFAULT_SPILL_PREVIEW_HALF_BYTES, preview_budget // 2)
    if half_budget < 64:
        return header

    head = _take_prefix_utf8(content, half_budget)
    tail = _take_suffix_utf8(content, half_budget)
    if len(head) + len(tail) >= len(content):
        return header
    head_bytes = len(head.encode("utf-8"))
    tail_bytes = len(tail.encode("utf-8"))
    omitted_bytes = max(0, raw_bytes - head_bytes - tail_bytes)
    return (
        f"{header}\n\n"
        f"--- HEAD ---\n{head}\n"
        f"--- ... omitted {omitted_bytes} bytes ... ---\n"
        f"--- TAIL ---\n{tail}"
    )


def _take_prefix_utf8(text: str, max_bytes: int) -> str:
    if max_bytes <= 0 or not text:
        return ""
    used = 0
    end = 0
    for index, ch in enumerate(text):
        width = len(ch.encode("utf-8"))
        if used + width > max_bytes:
            break
        used += width
        end = index + 1
    prefix = text[:end]
    newline = prefix.rfind("\n")
    if newline != -1 and newline >= int(len(prefix) * 0.7):
        return prefix[: newline + 1].rstrip("\r\n")
    return prefix


def _take_suffix_utf8(text: str, max_bytes: int) -> str:
    if max_bytes <= 0 or not text:
        return ""
    used = 0
    start = len(text)
    for index in range(len(text) - 1, -1, -1):
        width = len(text[index].encode("utf-8"))
        if used + width > max_bytes:
            break
        used += width
        start = index
    suffix = text[start:]
    newline = suffix.find("\n")
    if newline != -1 and newline <= int(len(suffix) * 0.3):
        return suffix[newline + 1 :].lstrip("\r\n")
    return suffix

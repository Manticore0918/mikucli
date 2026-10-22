"""Conservative provider-neutral prompt estimates, including tool schemas."""

from __future__ import annotations

import json

from mikucli.run_control import RunLimitExceeded


def estimate_prompt_tokens(messages: list[dict], tools: list[dict]) -> int:
    # A byte-based upper estimate avoids optimistic English-only chars/4 sizing.
    # Actual tokenization/framing is provider-specific; reserve additional overhead.
    payload = json.dumps({"messages": messages, "tools": tools}, ensure_ascii=False)
    return len(payload.encode("utf-8")) + 16 * len(messages) + 128


def output_reserve(window: int) -> int:
    return min(4096, max(1, window // 8))


def fits_prompt(messages: list[dict], tools: list[dict], window: int) -> bool:
    return estimate_prompt_tokens(messages, tools) + output_reserve(window) <= window


def check_prompt(messages: list[dict], tools: list[dict], window: int) -> None:
    if not fits_prompt(messages, tools, window):
        raise RunLimitExceeded(
            "context_window_exceeded",
            "Required context, protected task state, tool schemas, and output reserve exceed "
            "the configured context window. Increase --context-window-tokens or explicitly "
            "complete/supersede task records; protected state was retained.",
        )

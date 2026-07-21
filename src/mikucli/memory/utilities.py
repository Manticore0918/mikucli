from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from typing import Any

from .models import MemoryEntry, MemoryType


def dedupe_key(content: str) -> str:
    return " ".join(content.casefold().split())


def keywords(text: str) -> set[str]:
    return {term for term in re.findall(r"[A-Za-z0-9_]+", text.casefold()) if len(term) > 2}


def parse_timestamp(raw: str) -> datetime | None:
    try:
        value = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def chunk_memory_entries(entries: list[MemoryEntry], chunk_chars: int) -> list[str]:
    chunks: list[str] = []
    current: list[str] = []
    current_length = 0
    for entry in entries:
        text = entry_text(entry)
        if not text:
            continue
        separator_length = 2 if current else 0
        if current and current_length + separator_length + len(text) > chunk_chars:
            chunks.append("\n\n".join(current))
            current = []
            current_length = 0
        current.append(text)
        current_length += separator_length + len(text)
    if current:
        chunks.append("\n\n".join(current))
    return chunks


def entry_text(entry: MemoryEntry) -> str:
    text = entry_content(entry).strip()
    if not text:
        return ""
    return f"[{entry.type.value}] {text}"


def entry_content(entry: MemoryEntry) -> str:
    return (entry.content or messages_text(entry.messages)).strip()


def parse_json_string_list(raw: str) -> list[str]:
    cleaned = raw.strip()
    if not cleaned:
        return []
    start = cleaned.find("[")
    end = cleaned.rfind("]")
    if start != -1 and end != -1 and end > start:
        cleaned = cleaned[start : end + 1]
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        return []
    if not isinstance(parsed, list):
        return []
    return [item.strip() for item in parsed if isinstance(item, str) and item.strip()]


def recent_round_boundary(entries: list[MemoryEntry], retain_rounds: int) -> int:
    if retain_rounds == 0:
        return len(entries)
    seen = 0
    for index in range(len(entries) - 1, -1, -1):
        if is_real_user_turn(entries[index]):
            seen += 1
            if seen == retain_rounds:
                return index
    return 0


def is_real_user_turn(entry: MemoryEntry) -> bool:
    if entry.type != MemoryType.CONVERSATION or not entry.messages:
        return False
    message = entry.messages[0]
    if message.get("role") != "user":
        return False
    content = str(message.get("content") or "")
    return not content.startswith("Tool result for ")


def entry_has_tool_calls(entry: MemoryEntry) -> bool:
    return any(message.get("tool_calls") for message in entry.messages)


def messages_text(messages: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for message in messages:
        role = str(message.get("role") or "message")
        content = str(message.get("content") or "")
        if content:
            parts.append(f"{role}: {content}")
    return "\n".join(parts)

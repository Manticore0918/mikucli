from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal, Protocol


class MemoryType(str, Enum):
    CONVERSATION = "CONVERSATION"
    FACT = "FACT"
    SUMMARY = "SUMMARY"
    TOOL_RESULT = "TOOL_RESULT"


@dataclass
class MemoryEntry:
    type: MemoryType
    messages: list[dict[str, Any]]
    content: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: utc_now())


@dataclass(frozen=True)
class LongTermMemoryRecord:
    content: str
    created_at: str


@dataclass(frozen=True)
class LongTermMemorySaveResult:
    record: LongTermMemoryRecord
    saved: bool


@dataclass(frozen=True)
class RetrievedMemory:
    source: Literal["session", "long_term"]
    content: str
    created_at: str
    score: float


class ContextCompressor(Protocol):
    def compress(self, entries: list[MemoryEntry]) -> MemoryEntry | None: ...


class SummaryClient(Protocol):
    def chat(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        stream: bool = False,
    ) -> Any: ...


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()

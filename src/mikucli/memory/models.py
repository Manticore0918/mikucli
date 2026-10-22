from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import TYPE_CHECKING, Any, Literal, Protocol
from uuid import uuid4

if TYPE_CHECKING:
    from .compression import CompressionProposal
    from .ledger import ProtectedRecord


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
    entry_id: str = field(default_factory=lambda: uuid4().hex)


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
    def prepare(self, entries: list[MemoryEntry], protected: tuple[ProtectedRecord, ...] = (),
                *, scope_id: str = "") -> CompressionProposal | None: ...


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

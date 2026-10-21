"""Session and long-term memory capabilities."""

from .compression import MapReduceContextCompressor
from .long_term import (
    LongTermMemory,
    MemoryStorageWarning,
    default_long_term_memory_path,
)
from .models import (
    ContextCompressor,
    LongTermMemoryRecord,
    LongTermMemorySaveResult,
    MemoryEntry,
    MemoryType,
    RetrievedMemory,
    SummaryClient,
)
from .retrieval import MemoryRetrievalError, MemoryRetriever
from .session import SessionMemory
from .storage import MemoryStorageError


def token_usage_ratio(
    total_tokens: int | None, context_window_tokens: int
) -> float | None:
    if total_tokens is None:
        return None
    if context_window_tokens <= 0:
        raise ValueError("context_window_tokens must be positive")
    return total_tokens / context_window_tokens


__all__ = [
    "ContextCompressor",
    "LongTermMemory",
    "LongTermMemoryRecord",
    "LongTermMemorySaveResult",
    "MapReduceContextCompressor",
    "MemoryEntry",
    "MemoryRetrievalError",
    "MemoryRetriever",
    "MemoryStorageError",
    "MemoryStorageWarning",
    "MemoryType",
    "RetrievedMemory",
    "SessionMemory",
    "SummaryClient",
    "default_long_term_memory_path",
    "token_usage_ratio",
]

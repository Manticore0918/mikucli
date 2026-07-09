from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


JsonObject = dict[str, Any]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def duration_ms(started_at: str, ended_at: str) -> float | None:
    try:
        start = datetime.fromisoformat(started_at.replace("Z", "+00:00"))
        end = datetime.fromisoformat(ended_at.replace("Z", "+00:00"))
    except ValueError:
        return None
    return max((end - start).total_seconds() * 1000, 0.0)


@dataclass
class TraceRecord:
    trace_id: str
    run_id: str
    task_prompt: str
    workspace: str
    model: str
    session_mode: str
    started_at: str = field(default_factory=utc_now)
    ended_at: str | None = None
    status: str = "running"
    attributes: JsonObject = field(default_factory=dict)


@dataclass
class SpanEventRecord:
    trace_id: str
    span_id: str
    name: str
    at: str = field(default_factory=utc_now)
    attributes: JsonObject = field(default_factory=dict)


@dataclass
class SpanRecord:
    trace_id: str
    span_id: str
    parent_span_id: str | None
    name: str
    kind: str
    started_at: str = field(default_factory=utc_now)
    ended_at: str | None = None
    duration_ms: float | None = None
    status: str = "running"
    attributes: JsonObject = field(default_factory=dict)
    events: list[SpanEventRecord] = field(default_factory=list)


@dataclass
class MetricRecord:
    trace_id: str
    name: str
    value: int | float
    unit: str = ""
    at: str = field(default_factory=utc_now)
    attributes: JsonObject = field(default_factory=dict)

from __future__ import annotations

import math
import os
import traceback
from pathlib import Path
from threading import Lock
from typing import Any, Protocol

from .ids import new_span_id, new_trace_id
from .models import MetricRecord, SpanEventRecord, SpanRecord, TraceRecord, duration_ms, utc_now
from .store import LocalTraceStore, StoreMode


DEFAULT_STALE_TRACE_SECONDS = 3600.0


class TraceRecorder(Protocol):
    def start_trace(
        self,
        *,
        run_id: str,
        task_prompt: str,
        workspace: str,
        model: str,
        session_mode: str,
        attributes: dict[str, Any] | None = None,
    ) -> str: ...

    def end_trace(self, trace_id: str, *, status: str = "ok", attributes: dict[str, Any] | None = None) -> None: ...

    def start_span(
        self,
        *,
        trace_id: str,
        name: str,
        kind: str,
        parent_span_id: str | None = None,
        attributes: dict[str, Any] | None = None,
    ) -> str: ...

    def end_span(self, span_id: str, *, status: str = "ok", attributes: dict[str, Any] | None = None) -> None: ...
    def add_event(self, span_id: str, name: str, attributes: dict[str, Any] | None = None) -> None: ...

    def record_metric(
        self,
        trace_id: str,
        name: str,
        value: int | float,
        *,
        unit: str = "",
        attributes: dict[str, Any] | None = None,
    ) -> None: ...

    def diagnostics(self) -> list[str]: ...


class NoOpTraceRecorder:
    def start_trace(
        self,
        *,
        run_id: str,
        task_prompt: str,
        workspace: str,
        model: str,
        session_mode: str,
        attributes: dict[str, Any] | None = None,
    ) -> str:
        return ""

    def end_trace(self, trace_id: str, *, status: str = "ok", attributes: dict[str, Any] | None = None) -> None:
        return None

    def start_span(
        self,
        *,
        trace_id: str,
        name: str,
        kind: str,
        parent_span_id: str | None = None,
        attributes: dict[str, Any] | None = None,
    ) -> str:
        return ""

    def end_span(self, span_id: str, *, status: str = "ok", attributes: dict[str, Any] | None = None) -> None:
        return None

    def add_event(self, span_id: str, name: str, attributes: dict[str, Any] | None = None) -> None:
        return None

    def record_metric(
        self,
        trace_id: str,
        name: str,
        value: int | float,
        *,
        unit: str = "",
        attributes: dict[str, Any] | None = None,
    ) -> None:
        return None

    def diagnostics(self) -> list[str]:
        return []


class LocalTraceRecorder:
    def __init__(
        self,
        store: LocalTraceStore,
        *,
        stale_after_seconds: float | None = DEFAULT_STALE_TRACE_SECONDS,
    ) -> None:
        self.store = store
        self._traces: dict[str, TraceRecord] = {}
        self._spans: dict[str, SpanRecord] = {}
        self._span_trace_ids: dict[str, str] = {}
        self._diagnostics: list[str] = []
        self._lock = Lock()
        if stale_after_seconds is not None:
            self._best_effort(lambda: self.store.recover_stale_traces(stale_after_seconds=stale_after_seconds))

    def start_trace(
        self,
        *,
        run_id: str,
        task_prompt: str,
        workspace: str,
        model: str,
        session_mode: str,
        attributes: dict[str, Any] | None = None,
    ) -> str:
        trace = TraceRecord(
            trace_id=new_trace_id(),
            run_id=run_id,
            task_prompt=_capture_text(task_prompt, "MIKUCLI_OBS_CAPTURE_MESSAGES"),
            workspace=workspace,
            model=model,
            session_mode=session_mode,
            attributes=attributes or {},
        )
        with self._lock:
            self._traces[trace.trace_id] = trace
        self._best_effort(lambda: self.store.start_trace(trace))
        return trace.trace_id

    def end_trace(self, trace_id: str, *, status: str = "ok", attributes: dict[str, Any] | None = None) -> None:
        if not trace_id:
            return
        with self._lock:
            trace = self._traces.get(trace_id)
            if trace is None:
                return
            trace.ended_at = utc_now()
            trace.status = status
            if attributes:
                trace.attributes.update(attributes)
        self._best_effort(lambda: self.store.end_trace(trace))

    def start_span(
        self,
        *,
        trace_id: str,
        name: str,
        kind: str,
        parent_span_id: str | None = None,
        attributes: dict[str, Any] | None = None,
    ) -> str:
        if not trace_id:
            return ""
        span = SpanRecord(
            trace_id=trace_id,
            span_id=new_span_id(),
            parent_span_id=parent_span_id,
            name=name,
            kind=kind,
            attributes=attributes or {},
        )
        with self._lock:
            self._spans[span.span_id] = span
            self._span_trace_ids[span.span_id] = trace_id
        self._best_effort(lambda: self.store.start_span(span))
        return span.span_id

    def end_span(self, span_id: str, *, status: str = "ok", attributes: dict[str, Any] | None = None) -> None:
        if not span_id:
            return
        with self._lock:
            span = self._spans.get(span_id)
            if span is None:
                return
            span.ended_at = utc_now()
            span.duration_ms = duration_ms(span.started_at, span.ended_at)
            span.status = status
            if attributes:
                span.attributes.update(attributes)
        self._best_effort(lambda: self.store.end_span(span))

    def add_event(self, span_id: str, name: str, attributes: dict[str, Any] | None = None) -> None:
        if not span_id:
            return
        with self._lock:
            trace_id = self._span_trace_ids.get(span_id)
            if trace_id is None:
                return
            event = SpanEventRecord(trace_id=trace_id, span_id=span_id, name=name, attributes=attributes or {})
            span = self._spans.get(span_id)
            if span is not None:
                span.events.append(event)
        self._best_effort(lambda: self.store.add_event(event))

    def record_metric(
        self,
        trace_id: str,
        name: str,
        value: int | float,
        *,
        unit: str = "",
        attributes: dict[str, Any] | None = None,
    ) -> None:
        if not trace_id:
            return
        metric = MetricRecord(trace_id=trace_id, name=name, value=value, unit=unit, attributes=attributes or {})
        self._best_effort(lambda: self.store.record_metric(metric))

    def diagnostics(self) -> list[str]:
        return list(self._diagnostics)

    def _best_effort(self, action: Any) -> None:
        try:
            action()
        except Exception:
            self._diagnostics.append(traceback.format_exc(limit=1).strip())


def create_trace_recorder(workspace: Path) -> TraceRecorder:
    if not _env_bool("MIKUCLI_OBS_ENABLED"):
        return NoOpTraceRecorder()
    mode = os.environ.get("MIKUCLI_OBS_STORE", "sqlite").strip().casefold()
    if mode not in {"sqlite", "jsonl", "both"}:
        mode = "sqlite"
    store = LocalTraceStore(workspace / ".mikucli" / "observability", mode=mode)  # type: ignore[arg-type]
    stale_after_seconds = _env_nonnegative_float("MIKUCLI_OBS_STALE_AFTER_SECONDS")
    return LocalTraceRecorder(store, stale_after_seconds=stale_after_seconds)


def _env_bool(name: str) -> bool:
    value = os.environ.get(name, "").strip().casefold()
    return value in {"1", "true", "yes", "on"}


def _env_nonnegative_float(name: str, default: float = DEFAULT_STALE_TRACE_SECONDS) -> float:
    value = os.environ.get(name, "").strip()
    if not value:
        return default
    try:
        parsed = float(value)
    except ValueError:
        return default
    return parsed if math.isfinite(parsed) and parsed >= 0 else default


def _capture_text(text: str, env_name: str, *, limit: int = 500) -> str:
    mode = os.environ.get(env_name, "summary").strip().casefold()
    if mode == "off":
        return ""
    if mode == "full" or len(text) <= limit:
        return text
    return text[:limit].rstrip() + "\n... truncated ..."

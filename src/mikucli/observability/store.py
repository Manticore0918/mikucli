from __future__ import annotations

import sqlite3
from contextlib import closing
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Literal

from .eval_import import import_eval_report as import_eval_report_rows
from .models import MetricRecord, SpanEventRecord, SpanRecord, TraceRecord, duration_ms
from .schema import initialize_schema
from .serialization import json_object as _json_object
from .serialization import json_text as _json


StoreMode = Literal["sqlite", "jsonl", "both"]


class LocalTraceStore:
    """Persist trace records while delegating schema and eval-report concerns."""

    def __init__(self, root: Path, mode: StoreMode = "sqlite") -> None:
        if mode not in {"sqlite", "jsonl", "both"}:
            raise ValueError(f"unsupported observability store mode: {mode}")
        self.root = root
        self.mode = mode
        if self._writes_sqlite:
            self.sqlite_path.parent.mkdir(parents=True, exist_ok=True)
            self._init_sqlite()
        if self._writes_jsonl:
            self.trace_dir.mkdir(parents=True, exist_ok=True)

    @property
    def sqlite_path(self) -> Path:
        return self.root / "observability.sqlite3"

    @property
    def trace_dir(self) -> Path:
        return self.root / "traces"

    @property
    def _writes_sqlite(self) -> bool:
        return self.mode in {"sqlite", "both"}

    @property
    def _writes_jsonl(self) -> bool:
        return self.mode in {"jsonl", "both"}

    def start_trace(self, trace: TraceRecord) -> None:
        if self._writes_sqlite:
            with closing(self._connect()) as connection:
                connection.execute(
                    """
                    insert or replace into traces (
                        trace_id, run_id, task_prompt, workspace, model, session_mode,
                        started_at, ended_at, status, attributes_json
                    ) values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        trace.trace_id,
                        trace.run_id,
                        trace.task_prompt,
                        trace.workspace,
                        trace.model,
                        trace.session_mode,
                        trace.started_at,
                        trace.ended_at,
                        trace.status,
                        _json(trace.attributes),
                    ),
                )
                connection.commit()
        self._append_jsonl(trace.trace_id, "trace_started", asdict(trace))

    def end_trace(self, trace: TraceRecord) -> None:
        if self._writes_sqlite:
            with closing(self._connect()) as connection:
                connection.execute(
                    """
                    update traces
                    set ended_at = ?, status = ?, attributes_json = ?
                    where trace_id = ?
                    """,
                    (trace.ended_at, trace.status, _json(trace.attributes), trace.trace_id),
                )
                connection.commit()
        self._append_jsonl(trace.trace_id, "trace_ended", asdict(trace))

    def start_span(self, span: SpanRecord) -> None:
        if self._writes_sqlite:
            with closing(self._connect()) as connection:
                connection.execute(
                    """
                    insert or replace into spans (
                        span_id, trace_id, parent_span_id, name, kind,
                        started_at, ended_at, duration_ms, status, attributes_json
                    ) values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        span.span_id,
                        span.trace_id,
                        span.parent_span_id,
                        span.name,
                        span.kind,
                        span.started_at,
                        span.ended_at,
                        span.duration_ms,
                        span.status,
                        _json(span.attributes),
                    ),
                )
                connection.commit()
        self._append_jsonl(span.trace_id, "span_started", asdict(span))

    def end_span(self, span: SpanRecord) -> None:
        if self._writes_sqlite:
            with closing(self._connect()) as connection:
                connection.execute(
                    """
                    update spans
                    set ended_at = ?, duration_ms = ?, status = ?, attributes_json = ?
                    where span_id = ?
                    """,
                    (span.ended_at, span.duration_ms, span.status, _json(span.attributes), span.span_id),
                )
                connection.commit()
        self._append_jsonl(span.trace_id, "span_ended", asdict(span))

    def add_event(self, event: SpanEventRecord) -> None:
        if self._writes_sqlite:
            with closing(self._connect()) as connection:
                connection.execute(
                    """
                    insert into span_events (trace_id, span_id, name, at, attributes_json)
                    values (?, ?, ?, ?, ?)
                    """,
                    (event.trace_id, event.span_id, event.name, event.at, _json(event.attributes)),
                )
                connection.commit()
        self._append_jsonl(event.trace_id, "span_event", asdict(event))

    def record_metric(self, metric: MetricRecord) -> None:
        if self._writes_sqlite:
            with closing(self._connect()) as connection:
                connection.execute(
                    """
                    insert into metrics (trace_id, name, value, unit, at, attributes_json)
                    values (?, ?, ?, ?, ?, ?)
                    """,
                    (metric.trace_id, metric.name, metric.value, metric.unit, metric.at, _json(metric.attributes)),
                )
                connection.commit()
        self._append_jsonl(metric.trace_id, "metric", asdict(metric))

    def recover_stale_traces(
        self,
        *,
        stale_after_seconds: float,
        now: datetime | None = None,
    ) -> list[str]:
        if not self._writes_sqlite:
            return []
        recovered_at = now or datetime.now(timezone.utc)
        if recovered_at.tzinfo is None:
            recovered_at = recovered_at.replace(tzinfo=timezone.utc)
        recovered_at = recovered_at.astimezone(timezone.utc)
        recovered_at_text = recovered_at.isoformat()
        stale_before = (recovered_at - timedelta(seconds=max(stale_after_seconds, 0.0))).isoformat()
        recovery_records: list[tuple[str, str, dict[str, Any]]] = []
        recovered_trace_ids: list[str] = []

        with closing(self._connect()) as connection:
            connection.row_factory = sqlite3.Row
            connection.execute("begin immediate")
            traces = connection.execute(
                """
                select trace_id, attributes_json
                from traces
                where status = 'running'
                  and ended_at is null
                  and julianday(started_at) <= julianday(?)
                order by started_at
                """,
                (stale_before,),
            ).fetchall()
            for trace in traces:
                trace_id = str(trace["trace_id"])
                spans = connection.execute(
                    """
                    select span_id, started_at, attributes_json
                    from spans
                    where trace_id = ? and ended_at is null
                    order by started_at
                    """,
                    (trace_id,),
                ).fetchall()
                trace_attributes = _json_object(trace["attributes_json"])
                trace_attributes.update(
                    {
                        "recovery.reason": "startup_stale_trace",
                        "recovery.recovered_at": recovered_at_text,
                        "recovery.stale_after_seconds": stale_after_seconds,
                        "recovery.unfinished_span_count": len(spans),
                    }
                )
                updated = connection.execute(
                    """
                    update traces
                    set ended_at = ?, status = 'abandoned', attributes_json = ?
                    where trace_id = ? and status = 'running' and ended_at is null
                    """,
                    (recovered_at_text, _json(trace_attributes), trace_id),
                )
                if updated.rowcount != 1:
                    continue
                for span in spans:
                    span_attributes = _json_object(span["attributes_json"])
                    span_attributes.update(
                        {
                            "recovery.reason": "startup_stale_trace",
                            "recovery.recovered_at": recovered_at_text,
                            "recovery.ended_at_estimated": True,
                        }
                    )
                    connection.execute(
                        """
                        update spans
                        set ended_at = ?, duration_ms = ?, status = 'abandoned', attributes_json = ?
                        where span_id = ? and ended_at is null
                        """,
                        (
                            recovered_at_text,
                            duration_ms(str(span["started_at"]), recovered_at_text),
                            _json(span_attributes),
                            str(span["span_id"]),
                        ),
                    )
                    recovery_records.append(
                        (
                            trace_id,
                            "span_abandoned",
                            {
                                "span_id": str(span["span_id"]),
                                "ended_at": recovered_at_text,
                                "status": "abandoned",
                                "attributes": span_attributes,
                            },
                        )
                    )
                recovered_trace_ids.append(trace_id)
                recovery_records.append(
                    (
                        trace_id,
                        "trace_abandoned",
                        {
                            "trace_id": trace_id,
                            "ended_at": recovered_at_text,
                            "status": "abandoned",
                            "attributes": trace_attributes,
                        },
                    )
                )
            connection.commit()
        for trace_id, record_type, payload in recovery_records:
            self._append_jsonl(trace_id, record_type, payload)
        return recovered_trace_ids

    def import_eval_report(self, path: Path) -> None:
        if not self._writes_sqlite:
            return
        with closing(self._connect()) as connection:
            import_eval_report_rows(connection, path)

    def fetch_all(self, sql: str, parameters: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
        with closing(self._connect()) as connection:
            connection.row_factory = sqlite3.Row
            rows = connection.execute(sql, parameters).fetchall()
            return [dict(row) for row in rows]

    def fetch_one(self, sql: str, parameters: tuple[Any, ...] = ()) -> dict[str, Any] | None:
        rows = self.fetch_all(sql, parameters)
        return rows[0] if rows else None

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.sqlite_path)
        connection.execute("pragma journal_mode=wal")
        return connection

    def _init_sqlite(self) -> None:
        with closing(self._connect()) as connection:
            initialize_schema(connection)

    def _append_jsonl(self, trace_id: str, record_type: str, payload: dict[str, Any]) -> None:
        if not self._writes_jsonl:
            return
        path = self.trace_dir / f"{trace_id}.jsonl"
        with path.open("a", encoding="utf-8") as handle:
            handle.write(_json({"type": record_type, "payload": payload}) + "\n")

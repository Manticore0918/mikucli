from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from dataclasses import asdict
from pathlib import Path
from typing import Any, Literal

from .models import MetricRecord, SpanEventRecord, SpanRecord, TraceRecord


StoreMode = Literal["sqlite", "jsonl", "both"]


class LocalTraceStore:
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

    def import_eval_report(self, path: Path) -> None:
        if not self._writes_sqlite:
            return
        payload = json.loads(path.read_text(encoding="utf-8"))
        run_id = str(payload.get("run_id") or payload.get("run_group_id") or path.stem)
        run_group_id = str(payload.get("run_group_id") or run_id)
        summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
        results = payload.get("results") if isinstance(payload.get("results"), list) else []
        with closing(self._connect()) as connection:
            connection.execute(
                """
                insert or replace into runs (
                    run_id, run_group_id, started_at, model, total_cases, passed_cases,
                    success_rate, summary_json, report_path
                ) values (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    run_group_id,
                    str(payload.get("started_at") or ""),
                    str(payload.get("model") or ""),
                    int(summary.get("total_cases") or len(results)),
                    int(summary.get("passed_cases") or 0),
                    float(summary.get("success_rate") or 0.0),
                    _json(summary),
                    str(path),
                ),
            )
            old_case_ids = [
                row[0]
                for row in connection.execute("select case_result_id from eval_cases where run_id = ?", (run_id,)).fetchall()
            ]
            for case_result_id in old_case_ids:
                connection.execute("delete from eval_checks where case_result_id = ?", (case_result_id,))
                connection.execute("delete from tool_calls where case_result_id = ?", (case_result_id,))
                connection.execute("delete from approvals where case_result_id = ?", (case_result_id,))
            connection.execute("delete from eval_cases where run_id = ?", (run_id,))
            for result in results:
                if not isinstance(result, dict):
                    continue
                case_result_id = f"{run_id}:{result.get('case_id')}"
                connection.execute(
                    """
                    insert or replace into eval_cases (
                        case_result_id, run_id, run_group_id, case_id, task_id,
                        session_mode, passed, trace_id, workspace, run_log_path,
                        final_answer, changed_paths_json, metrics_json, failure_reasons_json
                    ) values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        case_result_id,
                        run_id,
                        str(result.get("run_group_id") or run_group_id),
                        str(result.get("case_id") or ""),
                        str(result.get("task_id") or ""),
                        str(result.get("session_mode") or ""),
                        1 if result.get("passed") else 0,
                        str(result.get("trace_id") or ""),
                        str(result.get("workspace") or ""),
                        str(result.get("run_log_path") or ""),
                        str(result.get("final_answer") or ""),
                        _json(result.get("changed_paths") or []),
                        _json(result.get("metrics") or {}),
                        _json(result.get("failure_reasons") or []),
                    ),
                )
                connection.execute("delete from eval_checks where case_result_id = ?", (case_result_id,))
                connection.execute("delete from tool_calls where case_result_id = ?", (case_result_id,))
                connection.execute("delete from approvals where case_result_id = ?", (case_result_id,))
                for group_name in ("check_results", "hallucination_results", "tool_correctness_results"):
                    for check in result.get(group_name) or []:
                        if not isinstance(check, dict):
                            continue
                        connection.execute(
                            """
                            insert into eval_checks (
                                case_result_id, name, category, passed, messages_json, evidence_json
                            ) values (?, ?, ?, ?, ?, ?)
                            """,
                            (
                                case_result_id,
                                str(check.get("name") or ""),
                                str(check.get("category") or group_name),
                                1 if check.get("passed") else 0,
                                _json(check.get("messages") or []),
                                _json(check.get("evidence") or {}),
                            ),
                        )
                for call in result.get("tool_calls") or []:
                    if not isinstance(call, dict):
                        continue
                    connection.execute(
                        """
                        insert into tool_calls (
                            case_result_id, name, arguments_json, ok, content, changed_paths_json
                        ) values (?, ?, ?, ?, ?, ?)
                        """,
                        (
                            case_result_id,
                            str(call.get("name") or ""),
                            _json(call.get("arguments") or {}),
                            1 if call.get("ok") else 0,
                            str(call.get("content") or ""),
                            _json(call.get("changed_paths") or []),
                        ),
                    )
                for approval in result.get("approvals") or []:
                    if not isinstance(approval, dict):
                        continue
                    connection.execute(
                        """
                        insert into approvals (
                            case_result_id, tool_name, risk_level, summary, details, approved
                        ) values (?, ?, ?, ?, ?, ?)
                        """,
                        (
                            case_result_id,
                            str(approval.get("tool_name") or ""),
                            str(approval.get("risk_level") or ""),
                            str(approval.get("summary") or ""),
                            str(approval.get("details") or ""),
                            1 if approval.get("approved") else 0,
                        ),
                    )
            connection.commit()

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
            connection.executescript(
                """
                create table if not exists traces (
                    trace_id text primary key,
                    run_id text not null,
                    task_prompt text not null,
                    workspace text not null,
                    model text not null,
                    session_mode text not null,
                    started_at text not null,
                    ended_at text,
                    status text not null,
                    attributes_json text not null
                );

                create table if not exists spans (
                    span_id text primary key,
                    trace_id text not null,
                    parent_span_id text,
                    name text not null,
                    kind text not null,
                    started_at text not null,
                    ended_at text,
                    duration_ms real,
                    status text not null,
                    attributes_json text not null,
                    foreign key(trace_id) references traces(trace_id)
                );

                create table if not exists span_events (
                    id integer primary key autoincrement,
                    trace_id text not null,
                    span_id text not null,
                    name text not null,
                    at text not null,
                    attributes_json text not null
                );

                create table if not exists metrics (
                    id integer primary key autoincrement,
                    trace_id text not null,
                    name text not null,
                    value real not null,
                    unit text not null,
                    at text not null,
                    attributes_json text not null
                );

                create table if not exists runs (
                    run_id text primary key,
                    run_group_id text not null,
                    started_at text not null,
                    model text not null,
                    total_cases integer not null,
                    passed_cases integer not null,
                    success_rate real not null,
                    summary_json text not null,
                    report_path text not null
                );

                create table if not exists eval_cases (
                    case_result_id text primary key,
                    run_id text not null,
                    run_group_id text not null,
                    case_id text not null,
                    task_id text not null,
                    session_mode text not null,
                    passed integer not null,
                    trace_id text not null,
                    workspace text not null,
                    run_log_path text not null,
                    final_answer text not null,
                    changed_paths_json text not null,
                    metrics_json text not null,
                    failure_reasons_json text not null
                );

                create table if not exists eval_checks (
                    id integer primary key autoincrement,
                    case_result_id text not null,
                    name text not null,
                    category text not null,
                    passed integer not null,
                    messages_json text not null,
                    evidence_json text not null
                );

                create table if not exists tool_calls (
                    id integer primary key autoincrement,
                    case_result_id text not null,
                    name text not null,
                    arguments_json text not null,
                    ok integer not null,
                    content text not null,
                    changed_paths_json text not null
                );

                create table if not exists approvals (
                    id integer primary key autoincrement,
                    case_result_id text not null,
                    tool_name text not null,
                    risk_level text not null,
                    summary text not null,
                    details text not null,
                    approved integer not null
                );

                create table if not exists regressions (
                    id integer primary key autoincrement,
                    base_run_id text not null,
                    head_run_id text not null,
                    case_id text not null,
                    category text not null,
                    details_json text not null
                );

                create index if not exists idx_traces_run_id on traces(run_id);
                create index if not exists idx_traces_started_at on traces(started_at);
                create index if not exists idx_spans_trace_id on spans(trace_id);
                create index if not exists idx_spans_parent_span_id on spans(parent_span_id);
                create index if not exists idx_spans_name on spans(name);
                create index if not exists idx_eval_cases_run_group_id on eval_cases(run_group_id);
                create index if not exists idx_eval_cases_case_id on eval_cases(case_id);
                create index if not exists idx_eval_cases_passed on eval_cases(passed);
                create index if not exists idx_eval_checks_case_result_id_passed on eval_checks(case_result_id, passed);
                """
            )
            connection.commit()

    def _append_jsonl(self, trace_id: str, record_type: str, payload: dict[str, Any]) -> None:
        if not self._writes_jsonl:
            return
        path = self.trace_dir / f"{trace_id}.jsonl"
        with path.open("a", encoding="utf-8") as handle:
            handle.write(_json({"type": record_type, "payload": payload}) + "\n")


def _json(value: Any) -> str:
    return json.dumps(_json_compatible(value), sort_keys=True, ensure_ascii=False)


def _json_compatible(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_compatible(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_compatible(item) for item in value]
    if isinstance(value, tuple):
        return [_json_compatible(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)

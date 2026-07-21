from __future__ import annotations

import sqlite3


def initialize_schema(connection: sqlite3.Connection) -> None:
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

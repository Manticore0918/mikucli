from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from .serialization import json_text


def import_eval_report(connection: sqlite3.Connection, path: Path) -> None:
    """Replace one persisted eval run and all of its dependent records."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    run_id = str(payload.get("run_id") or payload.get("run_group_id") or path.stem)
    run_group_id = str(payload.get("run_group_id") or run_id)
    summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
    results = payload.get("results") if isinstance(payload.get("results"), list) else []
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
            json_text(summary),
            str(path),
        ),
    )
    old_case_ids = [
        row[0]
        for row in connection.execute(
            "select case_result_id from eval_cases where run_id = ?",
            (run_id,),
        ).fetchall()
    ]
    for case_result_id in old_case_ids:
        _delete_case_children(connection, case_result_id)
    connection.execute("delete from eval_cases where run_id = ?", (run_id,))
    for result in results:
        if isinstance(result, dict):
            _insert_case(connection, result, run_id=run_id, run_group_id=run_group_id)
    connection.commit()


def _insert_case(
    connection: sqlite3.Connection,
    result: dict[str, object],
    *,
    run_id: str,
    run_group_id: str,
) -> None:
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
            json_text(result.get("changed_paths") or []),
            json_text(result.get("metrics") or {}),
            json_text(result.get("failure_reasons") or []),
        ),
    )
    _delete_case_children(connection, case_result_id)
    _insert_checks(connection, case_result_id, result)
    _insert_tool_calls(connection, case_result_id, result)
    _insert_approvals(connection, case_result_id, result)


def _delete_case_children(connection: sqlite3.Connection, case_result_id: str) -> None:
    connection.execute("delete from eval_checks where case_result_id = ?", (case_result_id,))
    connection.execute("delete from tool_calls where case_result_id = ?", (case_result_id,))
    connection.execute("delete from approvals where case_result_id = ?", (case_result_id,))


def _insert_checks(connection: sqlite3.Connection, case_result_id: str, result: dict[str, object]) -> None:
    for group_name in ("check_results", "hallucination_results", "tool_correctness_results"):
        for check in result.get(group_name) or []:  # type: ignore[union-attr]
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
                    json_text(check.get("messages") or []),
                    json_text(check.get("evidence") or {}),
                ),
            )


def _insert_tool_calls(connection: sqlite3.Connection, case_result_id: str, result: dict[str, object]) -> None:
    for call in result.get("tool_calls") or []:  # type: ignore[union-attr]
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
                json_text(call.get("arguments") or {}),
                1 if call.get("ok") else 0,
                str(call.get("content") or ""),
                json_text(call.get("changed_paths") or []),
            ),
        )


def _insert_approvals(connection: sqlite3.Connection, case_result_id: str, result: dict[str, object]) -> None:
    for approval in result.get("approvals") or []:  # type: ignore[union-attr]
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

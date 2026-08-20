from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

from .models import (
    ApprovalRecord,
    BenchmarkMetrics,
    BenchmarkResult,
    CheckResult,
    EstimatedSpend,
    EvalCost,
    EvalPrice,
    FailureReason,
    ToolCallRecord,
)
from ..report_validation import validate_report_payload


def atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def load_report(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not read benchmark report {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"benchmark report {path} must contain a JSON object")
    try:
        return validate_report_payload(payload)
    except ValueError as exc:
        raise ValueError(f"invalid benchmark report {path}: {exc}") from exc


def result_from_dict(raw: dict[str, Any]) -> BenchmarkResult:
    metrics_raw = _mapping(raw.get("metrics"))
    cost_raw = _mapping(metrics_raw.get("cost"))
    price_raw = metrics_raw.get("price")
    spend_raw = metrics_raw.get("estimated_spend")
    return BenchmarkResult(
        case_id=str(raw.get("case_id") or ""),
        task_id=str(raw.get("task_id") or ""),
        session_mode=str(raw.get("session_mode") or ""),
        passed=bool(raw.get("passed")),
        check_results=_checks(raw.get("check_results")),
        final_answer=str(raw.get("final_answer") or ""),
        changed_paths=[str(value) for value in _list(raw.get("changed_paths"))],
        tool_calls=[
            ToolCallRecord(
                name=str(value.get("name") or ""),
                arguments=_mapping(value.get("arguments")),
                ok=bool(value.get("ok")),
                content=str(value.get("content") or ""),
                changed_paths=[str(path) for path in _list(value.get("changed_paths"))],
            )
            for value in _dict_list(raw.get("tool_calls"))
        ],
        approvals=[
            ApprovalRecord(
                tool_name=str(value.get("tool_name") or ""),
                risk_level=str(value.get("risk_level") or ""),
                summary=str(value.get("summary") or ""),
                details=str(value.get("details") or ""),
                approved=bool(value.get("approved")),
            )
            for value in _dict_list(raw.get("approvals"))
        ],
        run_log_path=str(raw.get("run_log_path") or ""),
        workspace=str(raw.get("workspace") or ""),
        model=str(raw.get("model") or ""),
        elapsed_seconds=float(raw.get("elapsed_seconds") or 0.0),
        trial_index=int(raw.get("trial_index") or 1),
        trial_count=int(raw.get("trial_count") or 1),
        metrics=BenchmarkMetrics(
            tool_call_count=int(metrics_raw.get("tool_call_count") or 0),
            model_retries=int(metrics_raw.get("model_retries") or 0),
            step_retries=int(metrics_raw.get("step_retries") or 0),
            elapsed_seconds=float(metrics_raw.get("elapsed_seconds") or 0.0),
            cost=EvalCost(
                prompt_tokens=_optional_int(cost_raw.get("prompt_tokens")),
                completion_tokens=_optional_int(cost_raw.get("completion_tokens")),
                total_tokens=_optional_int(cost_raw.get("total_tokens")),
            ),
            price=_price(price_raw),
            estimated_spend=_spend(spend_raw),
        ),
        failure_reasons=[
            FailureReason(
                category=str(value.get("category") or ""),
                message=str(value.get("message") or ""),
                source=str(value.get("source") or ""),
            )
            for value in _dict_list(raw.get("failure_reasons"))
        ],
        trace_id=str(raw.get("trace_id") or ""),
        run_group_id=str(raw.get("run_group_id") or ""),
        baseline_run_id=str(raw.get("baseline_run_id") or ""),
        tool_correctness_results=_checks(raw.get("tool_correctness_results")),
        regression_status=str(raw.get("regression_status") or ""),
    )


def _checks(raw: Any) -> list[CheckResult]:
    return [
        CheckResult(
            name=str(value.get("name") or ""),
            passed=bool(value.get("passed")),
            messages=[str(message) for message in _list(value.get("messages"))],
            category=str(value.get("category") or "task_success"),
            evidence=_mapping(value.get("evidence")),
        )
        for value in _dict_list(raw)
    ]


def _price(raw: Any) -> EvalPrice | None:
    if not isinstance(raw, dict):
        return None
    return EvalPrice(
        prompt_token_price_per_million=_optional_float(raw.get("prompt_token_price_per_million")),
        completion_token_price_per_million=_optional_float(raw.get("completion_token_price_per_million")),
    )


def _spend(raw: Any) -> EstimatedSpend | None:
    if not isinstance(raw, dict):
        return None
    return EstimatedSpend(
        prompt=_optional_float(raw.get("prompt")),
        completion=_optional_float(raw.get("completion")),
        total=_optional_float(raw.get("total")),
    )


def _mapping(raw: Any) -> dict[str, Any]:
    return dict(raw) if isinstance(raw, dict) else {}


def _list(raw: Any) -> list[Any]:
    return list(raw) if isinstance(raw, list) else []


def _dict_list(raw: Any) -> list[dict[str, Any]]:
    return [dict(value) for value in _list(raw) if isinstance(value, dict)]


def _optional_int(raw: Any) -> int | None:
    return int(raw) if isinstance(raw, (int, float)) else None


def _optional_float(raw: Any) -> float | None:
    return float(raw) if isinstance(raw, (int, float)) else None

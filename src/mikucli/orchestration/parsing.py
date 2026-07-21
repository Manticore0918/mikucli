from __future__ import annotations

import json
import re
from typing import Any

from .models import ExecutionStep, ReviewDecision


def parse_execution_plan(raw: str) -> list[ExecutionStep]:
    plan = _extract_json_object(raw)
    raw_steps = plan.get("steps")
    if not isinstance(raw_steps, list) or not raw_steps:
        raise ValueError("planner output must contain a non-empty steps array")

    steps: list[ExecutionStep] = []
    seen_ids: set[str] = set()
    for index, raw_step in enumerate(raw_steps, start=1):
        if not isinstance(raw_step, dict):
            raise ValueError(f"step {index} must be an object")
        step_id = str(raw_step.get("id") or f"step-{index}").strip()
        if not step_id:
            raise ValueError(f"step {index} has an empty id")
        if step_id in seen_ids:
            raise ValueError(f"duplicate step id: {step_id}")
        task = str(raw_step.get("task") or raw_step.get("description") or "").strip()
        if not task:
            raise ValueError(f"{step_id} must include a task")
        depends_on = _string_list(
            raw_step.get("depends_on", raw_step.get("dependencies", raw_step.get("dependsOn", [])))
        )
        steps.append(
            ExecutionStep(
                id=step_id,
                title=str(raw_step.get("title") or "").strip(),
                task=task,
                depends_on=depends_on,
            )
        )
        seen_ids.add(step_id)

    known_ids = {step.id for step in steps}
    for step in steps:
        unknown = [dependency for dependency in step.depends_on if dependency not in known_ids]
        if unknown:
            raise ValueError(f"{step.id} depends on unknown step id(s): {', '.join(unknown)}")
    _validate_acyclic(steps)
    return steps


def parse_review_decision(raw: str) -> ReviewDecision:
    try:
        review = _extract_json_object(raw)
    except ValueError:
        return ReviewDecision(
            approved=False,
            summary=raw.strip(),
            issues=["Reviewer did not return valid JSON."],
        )
    approved_value = review.get("approved", review.get("passed"))
    issues = _string_list(review.get("issues", []))
    suggestions = _string_list(review.get("suggestions", []))
    legacy_problem = str(review.get("problems", review.get("feedback", "")) or "").strip()
    if legacy_problem and not issues:
        issues = [legacy_problem]
    return ReviewDecision(
        approved=_bool_value(approved_value),
        summary=str(review.get("summary") or "").strip(),
        issues=issues,
        suggestions=suggestions,
    )


def summarize_execution(steps: list[ExecutionStep]) -> str:
    lines = ["Execution summary:"]
    for step in steps:
        detail = step.result.strip() if step.status == "passed" else step.feedback.strip()
        if step.status == "skipped":
            detail = step.skipped_reason
        if not detail:
            detail = "No additional detail."
        lines.append(
            f"- {step.id} [{step.status}]"
            f"{f' via {step.assigned_worker}' if step.assigned_worker else ''}: {step.title or step.task}"
        )
        lines.append(f"  Result: {_compact(detail)}")
        if step.review_summary:
            lines.append(f"  Review: {_compact(step.review_summary)}")
    skipped = [step.id for step in steps if step.status == "skipped"]
    if skipped:
        lines.append("Skipped steps: " + ", ".join(skipped))
    return "\n".join(lines)


def _extract_json_object(raw: str) -> dict[str, Any]:
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("expected a JSON object")
    try:
        parsed = json.loads(cleaned[start : end + 1])
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON object: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ValueError("expected a JSON object")
    return parsed


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value.strip() else []
    if not isinstance(value, list):
        raise ValueError("dependencies must be a list of step ids")
    return [str(item).strip() for item in value if str(item).strip()]


def _bool_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().casefold() in {"true", "yes", "passed", "pass"}
    return bool(value)


def _validate_acyclic(steps: list[ExecutionStep]) -> None:
    step_by_id = {step.id: step for step in steps}
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(step_id: str) -> None:
        if step_id in visited:
            return
        if step_id in visiting:
            raise ValueError(f"dependency cycle includes {step_id}")
        visiting.add(step_id)
        for dependency in step_by_id[step_id].depends_on:
            visit(dependency)
        visiting.remove(step_id)
        visited.add(step_id)

    for step in steps:
        visit(step.id)


def _compact(text: str, limit: int = 500) -> str:
    compacted = " ".join(text.split())
    if len(compacted) <= limit:
        return compacted
    return compacted[:limit].rstrip() + "..."

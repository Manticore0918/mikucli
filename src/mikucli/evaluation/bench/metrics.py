from __future__ import annotations

import json
import math
import statistics
from pathlib import Path
from typing import Any

from .models import (
    BenchmarkResult,
    BenchmarkRunSummary,
    EstimatedSpend,
    EvalCost,
    EvalPrice,
    ToolCallRecord,
)


def summarize_results(
    results: list[BenchmarkResult],
    price: EvalPrice | None = None,
    *,
    stopped: bool = False,
    planned_trials: int | None = None,
    unique_cases: int | None = None,
    repeat_count: int = 1,
) -> BenchmarkRunSummary:
    total = len(results)
    passed = sum(1 for result in results if result.passed)
    cost = sum_costs([result.metrics.cost for result in results])
    elapsed = round(sum(result.metrics.elapsed_seconds for result in results), 3)
    elapsed_values = [result.metrics.elapsed_seconds for result in results]
    ci_low, ci_high = _wilson_interval(passed, total)
    planned = planned_trials if planned_trials is not None else total
    return BenchmarkRunSummary(
        total_cases=total,
        passed_cases=passed,
        success_rate=round(passed / total, 4) if total else 0.0,
        tool_call_count=sum(result.metrics.tool_call_count for result in results),
        model_retries=sum(result.metrics.model_retries for result in results),
        step_retries=sum(result.metrics.step_retries for result in results),
        elapsed_seconds=elapsed,
        cost=cost,
        price=price,
        estimated_spend=estimate_spend(cost, price),
        stopped=stopped,
        planned_trials=planned,
        completed_trials=total,
        incomplete_trials=max(0, planned - total),
        unique_cases=unique_cases if unique_cases is not None else len({result.case_id for result in results}),
        repeat_count=repeat_count,
        median_elapsed_seconds=round(statistics.median(elapsed_values), 3) if elapsed_values else None,
        p95_elapsed_seconds=round(_percentile(elapsed_values, 0.95), 3) if elapsed_values else None,
        pass_rate_ci95_low=ci_low,
        pass_rate_ci95_high=ci_high,
    )


def cost_from_usage(events: list[Any]) -> EvalCost:
    return EvalCost(
        prompt_tokens=_complete_sum([event.prompt_tokens for event in events]),
        completion_tokens=_complete_sum([event.completion_tokens for event in events]),
        total_tokens=_complete_sum([event.total_tokens for event in events]),
    )


def sum_costs(costs: list[EvalCost]) -> EvalCost:
    return EvalCost(
        prompt_tokens=_complete_sum([cost.prompt_tokens for cost in costs]),
        completion_tokens=_complete_sum([cost.completion_tokens for cost in costs]),
        total_tokens=_complete_sum([cost.total_tokens for cost in costs]),
    )


def _complete_sum(values: list[int | None]) -> int | None:
    if not values or any(value is None for value in values):
        return None
    return sum(value for value in values if value is not None)


def estimate_spend(cost: EvalCost, price: EvalPrice | None) -> EstimatedSpend | None:
    if price is None:
        return None
    prompt = _component_spend(cost.prompt_tokens, price.prompt_token_price_per_million)
    completion = _component_spend(cost.completion_tokens, price.completion_token_price_per_million)
    total = round(prompt + completion, 8) if prompt is not None and completion is not None else None
    return EstimatedSpend(prompt=prompt, completion=completion, total=total)


def model_retries(tool_calls: list[ToolCallRecord], final_answer: str) -> int:
    retries = sum(1 for call in tool_calls if not call.ok)
    if final_answer == "Stopped because the session reached the maximum tool loop depth.":
        retries += 1
    return retries


def step_retries_from_log(path: Path) -> int:
    attempts_by_step: dict[str, int] = {}
    for event in read_log_events(path):
        if event.get("type") != "step_worker_result":
            continue
        step_id = str(event.get("step_id") or "")
        if not step_id:
            continue
        try:
            attempt = int(event.get("attempt") or 0)
        except (TypeError, ValueError):
            continue
        attempts_by_step[step_id] = max(attempts_by_step.get(step_id, 0), attempt)
    return sum(max(0, attempts - 1) for attempts in attempts_by_step.values())


def trace_id_from_run_log(path: Path) -> str:
    payload = read_log_payload(path)
    metadata = payload.get("metadata")
    if isinstance(metadata, dict):
        return str(metadata.get("trace_id") or "")
    return ""


def read_log_events(path: Path) -> list[dict[str, Any]]:
    payload = read_log_payload(path)
    events = payload.get("events")
    return events if isinstance(events, list) else []


def read_log_payload(path: Path) -> dict[str, Any]:
    if not path or not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _component_spend(tokens: int | None, price_per_million: float | None) -> float | None:
    if tokens is None or price_per_million is None:
        return None
    return round(tokens * price_per_million / 1_000_000, 8)


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def _wilson_interval(successes: int, total: int) -> tuple[float | None, float | None]:
    if total <= 0:
        return None, None
    z = 1.959963984540054
    proportion = successes / total
    denominator = 1 + z * z / total
    centre = (proportion + z * z / (2 * total)) / denominator
    margin = z * math.sqrt(
        proportion * (1 - proportion) / total + z * z / (4 * total * total)
    ) / denominator
    return round(max(0.0, centre - margin), 4), round(min(1.0, centre + margin), 4)

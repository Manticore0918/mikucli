from __future__ import annotations

import json
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
) -> BenchmarkRunSummary:
    total = len(results)
    passed = sum(1 for result in results if result.passed)
    cost = sum_costs([result.metrics.cost for result in results])
    elapsed = round(sum(result.metrics.elapsed_seconds for result in results), 3)
    agent_latency = round(sum(result.metrics.agent_latency_seconds for result in results), 3)
    llm_latency = round(sum(result.metrics.llm_latency_seconds for result in results), 3)
    return BenchmarkRunSummary(
        total_cases=total,
        passed_cases=passed,
        success_rate=round(passed / total, 4) if total else 0.0,
        tool_call_count=sum(result.metrics.tool_call_count for result in results),
        model_retries=sum(result.metrics.model_retries for result in results),
        step_retries=sum(result.metrics.step_retries for result in results),
        elapsed_seconds=elapsed,
        agent_latency_seconds=agent_latency,
        llm_latency_seconds=llm_latency,
        cost=cost,
        price=price,
        estimated_spend=estimate_spend(cost, price),
        stopped=stopped,
    )


def cost_from_usage(events: list[Any]) -> EvalCost:
    prompt_values = [event.prompt_tokens for event in events if event.prompt_tokens is not None]
    completion_values = [event.completion_tokens for event in events if event.completion_tokens is not None]
    total_values = [event.total_tokens for event in events if event.total_tokens is not None]
    return EvalCost(
        prompt_tokens=sum(prompt_values) if prompt_values else None,
        completion_tokens=sum(completion_values) if completion_values else None,
        total_tokens=sum(total_values) if total_values else None,
    )


def sum_costs(costs: list[EvalCost]) -> EvalCost:
    prompt_values = [cost.prompt_tokens for cost in costs if cost.prompt_tokens is not None]
    completion_values = [cost.completion_tokens for cost in costs if cost.completion_tokens is not None]
    total_values = [cost.total_tokens for cost in costs if cost.total_tokens is not None]
    return EvalCost(
        prompt_tokens=sum(prompt_values) if prompt_values else None,
        completion_tokens=sum(completion_values) if completion_values else None,
        total_tokens=sum(total_values) if total_values else None,
    )


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

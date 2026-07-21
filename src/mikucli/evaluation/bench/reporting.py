from __future__ import annotations

from pathlib import Path

from .models import BenchmarkResult, BenchmarkRunSummary, EstimatedSpend, EvalCost


def markdown_report(
    run_id: str,
    model: str,
    summary: BenchmarkRunSummary,
    results: list[BenchmarkResult],
    json_path: Path,
) -> str:
    lines = [
        f"# Benchmark Run {run_id}",
        "",
        f"- Model: `{model}`",
        f"- JSON results: `{json_path}`",
        f"- Success rate: {_fmt_percent(summary.success_rate)} ({summary.passed_cases}/{summary.total_cases})",
        f"- Stopped: {'yes' if summary.stopped else 'no'}",
        f"- Tool calls: {summary.tool_call_count}",
        f"- Model retries: {summary.model_retries}",
        f"- Step retries: {summary.step_retries}",
        f"- Total latency: {summary.elapsed_seconds:.3f}s",
        f"- Agent latency: {summary.agent_latency_seconds:.3f}s",
        f"- LLM latency: {summary.llm_latency_seconds:.3f}s",
        f"- Cost: {_fmt_cost(summary.cost)}",
        f"- Estimated spend: {_fmt_spend(summary.estimated_spend)}",
        "",
        "## Cases",
        "",
        "| Status | Case | Mode | Tool calls | Model retries | Step retries | Total latency | Agent latency | LLM latency | Cost | Spend |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for result in results:
        status = "PASS" if result.passed else "FAIL"
        lines.append(
            "| "
            + " | ".join(
                [
                    status,
                    _md_cell(result.case_id),
                    _md_cell(result.session_mode),
                    str(result.metrics.tool_call_count),
                    str(result.metrics.model_retries),
                    str(result.metrics.step_retries),
                    f"{result.metrics.elapsed_seconds:.3f}s",
                    f"{result.metrics.agent_latency_seconds:.3f}s",
                    f"{result.metrics.llm_latency_seconds:.3f}s",
                    _md_cell(_fmt_cost(result.metrics.cost)),
                    _md_cell(_fmt_spend(result.metrics.estimated_spend)),
                ]
            )
            + " |"
        )
    failed = [result for result in results if result.failure_reasons]
    if failed:
        lines.extend(["", "## Failure Reasons", ""])
        for result in failed:
            lines.extend([f"### {result.case_id}", ""])
            for reason in result.failure_reasons:
                lines.append(f"- `{reason.category}` from `{reason.source}`: {reason.message}")
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _fmt_cost(cost: EvalCost) -> str:
    prompt = _fmt_optional_int(cost.prompt_tokens)
    completion = _fmt_optional_int(cost.completion_tokens)
    total = _fmt_optional_int(cost.total_tokens)
    return f"prompt={prompt}, completion={completion}, total={total}"


def _fmt_spend(spend: EstimatedSpend | None) -> str:
    if spend is None:
        return "unknown"
    return f"prompt={_fmt_optional_float(spend.prompt)}, completion={_fmt_optional_float(spend.completion)}, total={_fmt_optional_float(spend.total)}"


def _fmt_percent(value: float) -> str:
    return f"{value * 100:.1f}%"


def _fmt_optional_int(value: int | None) -> str:
    return "unknown" if value is None else str(value)


def _fmt_optional_float(value: float | None) -> str:
    return "unknown" if value is None else f"{value:.8f}".rstrip("0").rstrip(".")


def _md_cell(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")

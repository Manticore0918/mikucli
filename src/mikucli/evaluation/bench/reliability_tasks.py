"""Offline workflow contracts; these measure the runner, not model intelligence."""

from __future__ import annotations

from .catalog import validate_benchmark_tasks
from .models import BenchmarkContext, BenchmarkTask, SessionMode, TaskSetup

# id, expected runtime outcome, reason, expected HTTP attempts
CONTRACTS = (
    ("provider_retry_recovery", "succeeded", "completed", 4),
    ("provider_exhaustion", "failed", "provider_error", 3),
    ("provider_invalid_response", "failed", "invalid_provider_response", 1),
    ("provider_request_timeout", "failed", "provider_error", 2),
    ("cancel_request", "stopped", "user_request", 1),
    ("cancel_backoff", "stopped", "user_request", 1),
    ("token_limit", "limit_reached", "token_limit", 1),
    ("usage_unavailable", "limit_reached", "usage_unavailable", 1),
    ("spend_limit", "limit_reached", "spend_limit", 1),
    ("duration_limit", "limit_reached", "duration_limit", 1),
    ("output_limit", "limit_reached", "output_limit", 1),
    ("approval_denial", "failed", "approval_denied", 2),
    ("cancel_approval", "stopped", "user_request", 1),
    ("cancel_tool", "stopped", "user_request", 1),
    ("team_worker_failure", "failed", "provider_error", 2),
    ("team_reviewer_failure", "failed", "provider_error", 3),
    ("team_shared_budget", "limit_reached", "token_limit", 2),
    ("durable_restart_recovery", "succeeded", "completed", 1),
    ("durable_concurrent_writers", "succeeded", "completed", 1),
    ("durable_interrupted_transaction", "succeeded", "completed", 1),
    ("mcp_queued_cancellation", "limit_reached", "duration_limit", 1),
)


def scenario_invariants(context: BenchmarkContext) -> list[str]:
    scenario = context.setup.metadata.get("scenario")
    if scenario is None:
        return ["offline contract requires the reliability runner"]
    errors = list(scenario.errors)
    if scenario.attempts != scenario.expected_attempts:
        errors.append(
            f"expected {scenario.expected_attempts} HTTP attempts, got {scenario.attempts}"
        )
    if (context.workspace / "late.txt").exists():
        errors.append("work executed after a denial, cancellation, or budget limit")
    if scenario.tools is not None and getattr(
        scenario.tools, "_active_processes", set()
    ):
        errors.append("a tool process was left running")
    return errors


scenario_invariants.category = "reliability_contract"


def reliability_tasks() -> list[BenchmarkTask]:
    return validate_benchmark_tasks(
        [
            BenchmarkTask(
                id=name,
                title=name.replace("_", " ").title(),
                prompt=f"Exercise the offline runner contract: {name}.",
                setup=lambda _: TaskSetup(),
                checks=(scenario_invariants,),
                session_modes=(SessionMode.BUILT_IN_MULTI_AGENT,)
                if name.startswith("team_")
                else (SessionMode.BUILT_IN_SINGLE_AGENT,),
                suite="reliability",
                capabilities=frozenset({"runtime_contract"}),
                expected_status=status,
                expected_reason=reason,
                expected_tool_failures=frozenset(
                    {"write_file", "run_shell", "remote_tool"}
                )
                if status != "succeeded"
                else frozenset(),
            )
            for name, status, reason, _ in CONTRACTS
        ]
    )

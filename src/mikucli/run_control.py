"""Per-turn limits shared by the orchestrator, workers, and compression calls."""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from functools import wraps
from threading import Lock
from typing import Any


class RunLimitExceeded(RuntimeError):
    def __init__(self, reason: str, message: str) -> None:
        super().__init__(message)
        self.reason = reason


@dataclass(frozen=True)
class RunLimits:
    max_duration_seconds: float = 900
    max_tokens: int | None = None
    max_spend: float | None = None
    max_output_bytes: int = 1_048_576
    prompt_price_per_million: float | None = None
    completion_price_per_million: float | None = None

    def __post_init__(self) -> None:
        for value in (self.max_tokens, self.max_output_bytes):
            if value is not None and (
                not isinstance(value, int) or isinstance(value, bool)
            ):
                raise ValueError("token and output limits must be integer counts")
        for name in (
            "max_duration_seconds",
            "max_tokens",
            "max_spend",
            "max_output_bytes",
        ):
            value = getattr(self, name)
            if value is not None and (not math.isfinite(value) or value <= 0):
                raise ValueError(f"{name} must be finite and positive")
        for value in (self.prompt_price_per_million, self.completion_price_per_million):
            if value is not None and (not math.isfinite(value) or value < 0):
                raise ValueError("token prices must be finite and non-negative")
        if self.max_spend is not None and (
            self.prompt_price_per_million is None
            or self.completion_price_per_million is None
        ):
            raise ValueError(
                "--max-spend requires both prompt and completion token prices"
            )


class RunBudget:
    def __init__(self, limits: RunLimits) -> None:
        self.limits = limits
        self.started = time.monotonic()
        self.tokens = 0
        self.spend = 0.0
        self.output_bytes = 0
        self.model_calls = 0
        self.http_attempts = 0
        self.usage_known = True
        self.spend_known = True
        self.failure: RunLimitExceeded | None = None
        self._lock = Lock()
        self._model_lock = Lock()

    def fail(self, reason: str, message: str) -> None:
        with self._lock:
            if self.failure is None:
                self.failure = RunLimitExceeded(reason, message)
            error = self.failure
        raise error

    def check(self) -> None:
        if self.failure is not None:
            raise self.failure
        if self.remaining_seconds() <= 0:
            self.fail(
                "duration_limit", "Stopped because the task duration limit was reached."
            )

    def remaining_seconds(self) -> float:
        return max(
            0.0, self.limits.max_duration_seconds - (time.monotonic() - self.started)
        )

    def output_allowance(self, count: int) -> int:
        with self._lock:
            allowed = min(
                count, max(0, self.limits.max_output_bytes - self.output_bytes)
            )
            self.output_bytes += allowed
            if allowed < count and self.failure is None:
                self.failure = RunLimitExceeded(
                    "output_limit",
                    "Stopped because the task tool-output limit was reached.",
                )
            return allowed

    def account(self, usage: Any) -> None:
        total = usage.total_tokens
        if (
            total is None
            and usage.prompt_tokens is not None
            and usage.completion_tokens is not None
        ):
            total = usage.prompt_tokens + usage.completion_tokens
        if (
            total is not None
            and usage.prompt_tokens is not None
            and usage.completion_tokens is not None
        ):
            total = max(total, usage.prompt_tokens + usage.completion_tokens)
        with self._lock:
            self.usage_known = self.usage_known and total is not None
            self.tokens += total or 0
            complete = (
                usage.prompt_tokens is not None and usage.completion_tokens is not None
            )
            self.spend_known = self.spend_known and complete
            if (
                complete
                and self.limits.prompt_price_per_million is not None
                and self.limits.completion_price_per_million is not None
            ):
                estimate = (
                    usage.prompt_tokens
                    / 1_000_000
                    * self.limits.prompt_price_per_million
                    + usage.completion_tokens
                    / 1_000_000
                    * self.limits.completion_price_per_million
                )
                self.spend_known = self.spend_known and math.isfinite(
                    self.spend + estimate
                )
                if self.spend_known:
                    self.spend += estimate
        if (self.limits.max_tokens is not None and total is None) or (
            self.limits.max_spend is not None and not self.spend_known
        ):
            self.fail(
                "usage_unavailable",
                "Provider usage is missing or unrepresentable; the configured token/spend limit cannot be enforced.",
            )
        if self.limits.max_tokens is not None and self.tokens >= self.limits.max_tokens:
            self.fail(
                "token_limit", "Stopped because the reported token limit was reached."
            )
        if self.limits.max_spend is not None and self.spend >= self.limits.max_spend:
            self.fail(
                "spend_limit", "Stopped because the estimated spend limit was reached."
            )
        self.check()

    @contextmanager
    def model_slot(self):
        limited = (
            self.limits.max_tokens is not None or self.limits.max_spend is not None
        )
        if limited:
            while not self._model_lock.acquire(timeout=0.05):
                check_active()
        try:
            self.check()
            yield
        finally:
            if limited:
                self._model_lock.release()

    def metrics(self) -> dict[str, Any]:
        return {
            "elapsed_seconds": round(time.monotonic() - self.started, 3),
            "reported_tokens": self.tokens if self.usage_known else None,
            "estimated_spend": self.spend
            if self.spend_known
            and self.limits.prompt_price_per_million is not None
            and self.limits.completion_price_per_million is not None
            else None,
            "tool_output_bytes": self.output_bytes,
            "model_calls": self.model_calls,
            "http_attempts": self.http_attempts,
        }

    def record_http_attempt(self) -> None:
        with self._lock:
            self.http_attempts += 1


_budget: ContextVar[RunBudget | None] = ContextVar("mikucli_run_budget", default=None)


def current_budget() -> RunBudget | None:
    return _budget.get()


def check_active() -> None:
    from .cancellation import current_stop_requested, raise_if_stop_requested

    raise_if_stop_requested(current_stop_requested())


def managed_turn(function: Callable[..., Any]) -> Callable[..., Any]:
    @wraps(function)
    def wrapped(self: Any, *args: Any, **kwargs: Any) -> Any:
        existing = current_budget()
        token = _budget.set(
            existing or RunBudget(getattr(self, "run_limits", RunLimits()))
        )
        try:
            return function(self, *args, **kwargs)
        finally:
            _budget.reset(token)

    return wrapped


def call_chat(client: Any, **kwargs: Any) -> Any:
    budget = current_budget()
    if budget is None:
        return client.chat(**kwargs)
    with budget.model_slot():
        with budget._lock:
            budget.model_calls += 1
        try:
            response = client.chat(**kwargs)
        except Exception as exc:
            if getattr(exc, "token_usage", None) is not None:
                budget.account(exc.token_usage)
            else:
                budget.usage_known = budget.spend_known = False
            raise
        budget.account(response.token_usage)
        return response

"""Bounded, cancellable HTTP requests for model and embedding services."""

from __future__ import annotations

import asyncio
import math
import random
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from typing import Any

import httpx

from .run_control import check_active, current_budget


@dataclass(frozen=True)
class RequestPolicy:
    timeout_seconds: float = 120
    max_retries: int = 2
    max_retry_delay: float = 30
    max_response_bytes: int = 8_388_608

    def __post_init__(self) -> None:
        if not math.isfinite(self.timeout_seconds) or self.timeout_seconds <= 0:
            raise ValueError("request timeout must be finite and positive")
        if not isinstance(self.max_retries, int) or not 0 <= self.max_retries <= 10:
            raise ValueError("max retries must be between 0 and 10")
        if not math.isfinite(self.max_retry_delay) or self.max_retry_delay < 0:
            raise ValueError("retry delay must be finite and non-negative")
        if self.max_response_bytes <= 0:
            raise ValueError("response size limit must be positive")


class ProviderRequestError(RuntimeError):
    reason = "provider_error"


class ModelResponseError(RuntimeError):
    reason = "invalid_provider_response"


async def cancellable(awaitable: Any, timeout: float) -> Any:
    task = asyncio.create_task(awaitable)
    deadline = time.monotonic() + timeout
    try:
        while True:
            check_active()
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise httpx.ReadTimeout("request deadline exceeded")
            done, _ = await asyncio.wait({task}, timeout=min(0.05, remaining))
            if task in done:
                result = await task
                check_active()
                return result
    finally:
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)


def retry_after(value: str) -> float | None:
    try:
        seconds = float(value)
    except ValueError:
        try:
            date = parsedate_to_datetime(value)
            if date.tzinfo is None:
                date = date.replace(tzinfo=UTC)
            seconds = (date - datetime.now(UTC)).total_seconds()
        except (ValueError, TypeError, OverflowError):
            return None
    return max(0.0, seconds) if math.isfinite(seconds) else None


async def request_json(
    url: str,
    *,
    policy: RequestPolicy,
    method: str = "POST",
    payload: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
    transport: httpx.AsyncBaseTransport | None = None,
) -> tuple[Any, int]:
    async with httpx.AsyncClient(
        transport=transport, timeout=policy.timeout_seconds
    ) as client:

        async def send() -> tuple[int, httpx.Headers, bytes]:
            async with client.stream(
                method, url, json=payload, headers=headers
            ) as response:
                content = bytearray()
                async for chunk in response.aiter_bytes():
                    check_active()
                    if len(content) + len(chunk) > policy.max_response_bytes:
                        raise ModelResponseError(
                            "Provider response exceeded the response-size limit."
                        )
                    content.extend(chunk)
                return response.status_code, response.headers, bytes(content)

        for attempt in range(policy.max_retries + 1):
            check_active()
            budget = current_budget()
            if budget is not None:
                budget.record_http_attempt()
            delay = None
            try:
                status, response_headers, body = await cancellable(
                    send(), policy.timeout_seconds
                )
            except httpx.UnsupportedProtocol as exc:
                raise ProviderRequestError(
                    "Provider endpoint uses an unsupported protocol."
                ) from exc
            except httpx.TransportError as exc:
                failure = f"Provider request failed ({type(exc).__name__}); check connectivity and request timeout."
            else:
                if 200 <= status < 300:
                    import json

                    try:
                        return json.loads(body), attempt + 1
                    except (ValueError, UnicodeError) as exc:
                        raise ModelResponseError(
                            "Provider returned invalid JSON."
                        ) from exc
                failure = f"Provider HTTP {status}; check credentials, model, endpoint, or service availability."
                if status not in {408, 429, 500, 502, 503, 504}:
                    raise ProviderRequestError(failure)
                delay = retry_after(response_headers.get("Retry-After", ""))
            if attempt == policy.max_retries:
                raise ProviderRequestError(
                    f"{failure} Exhausted {attempt + 1} attempt(s)."
                )
            if delay is not None and delay > policy.max_retry_delay:
                raise ProviderRequestError(
                    f"{failure} Retry-After exceeds the configured retry-delay limit."
                )
            if delay is None:
                delay = min(
                    policy.max_retry_delay, 0.5 * 2**attempt * random.uniform(0.5, 1.0)
                )
            await cancellable(asyncio.sleep(delay), delay + 1)
    raise AssertionError("unreachable request state")

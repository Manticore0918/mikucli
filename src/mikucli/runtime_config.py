from __future__ import annotations

import os
from typing import Any
from urllib.parse import urlsplit

from .http_requests import RequestPolicy
from .run_control import RunLimits


def validate_endpoint(value: str, name: str) -> None:
    parsed = urlsplit(value)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError(f"{name} must be an absolute HTTP or HTTPS URL")


def runtime_settings(
    files: dict[str, str], overrides: dict[str, Any] | None
) -> tuple[RequestPolicy, RunLimits]:
    overrides = overrides or {}

    def setting(key: str, env: str, convert: Any, default: Any = None) -> Any:
        value = overrides.get(key)
        if value is None:
            value = os.environ.get(env) or files.get(env)
        if value is None or value == "":
            return default
        try:
            return convert(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid value for {env}") from exc

    policy = RequestPolicy(
        timeout_seconds=setting(
            "request_timeout", "MIKUCLI_REQUEST_TIMEOUT_SECONDS", float, 120
        ),
        max_retries=setting("max_retries", "MIKUCLI_MAX_RETRIES", int, 2),
        max_retry_delay=setting(
            "retry_max_delay", "MIKUCLI_RETRY_MAX_DELAY_SECONDS", float, 30
        ),
    )
    limits = RunLimits(
        max_duration_seconds=setting(
            "max_duration", "MIKUCLI_MAX_DURATION_SECONDS", float, 900
        ),
        max_tokens=setting("max_tokens", "MIKUCLI_MAX_TOKENS", int),
        max_spend=setting("max_spend", "MIKUCLI_MAX_SPEND", float),
        max_output_bytes=setting(
            "max_output_bytes", "MIKUCLI_MAX_OUTPUT_BYTES", int, 1_048_576
        ),
        prompt_price_per_million=setting(
            "prompt_token_price_per_million",
            "MIKUCLI_PROMPT_TOKEN_PRICE_PER_MILLION",
            float,
        ),
        completion_price_per_million=setting(
            "completion_token_price_per_million",
            "MIKUCLI_COMPLETION_TOKEN_PRICE_PER_MILLION",
            float,
        ),
    )
    return policy, limits

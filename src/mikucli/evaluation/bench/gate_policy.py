"""Strict, catalog-validated release policy. Typos never silently disable a gate."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .evidence import identifier, number
from .suites import benchmark_cases

SUITES = ("core", "stress", "realworld", "reliability")
POLICY_FIELDS = {
    "schema_version",
    "name",
    "minimum_repeat_count",
    "require_baseline",
    "require_usage",
    "require_clean_revision",
    "maximum_pass_rate_drop",
    "maximum_median_latency_ratio",
    "maximum_p95_latency_ratio",
    "maximum_median_tokens_ratio",
    "maximum_median_spend_ratio",
    "cases",
}
CASE_FIELDS = {
    "minimum_pass_rate",
    "minimum_ci95_low",
    "maximum_median_latency_seconds",
    "maximum_p95_latency_seconds",
    "maximum_median_tokens",
    "maximum_median_spend",
}


def strict_json(path: Path) -> dict[str, Any]:
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    def invalid_constant(value):
        raise ValueError(f"non-finite JSON value: {value}")

    payload = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=unique,
        parse_constant=invalid_constant,
    )
    if not isinstance(payload, dict):
        raise ValueError("expected a JSON object")
    return payload


def catalog():
    return {case.id: case for suite in SUITES for case in benchmark_cases(suite)}


def load_policy(name_or_path: str) -> dict[str, Any]:
    path = Path(name_or_path)
    if name_or_path in {"offline", "canary", "release"}:
        path = Path(__file__).parent / "fixtures" / f"policy-{name_or_path}.json"
    return validate_policy(strict_json(path))


def validate_policy(policy: dict[str, Any]) -> dict[str, Any]:
    unknown = set(policy) - POLICY_FIELDS
    if unknown:
        raise ValueError("unknown policy fields: " + ", ".join(sorted(unknown)))
    if type(policy.get("schema_version")) is not int or policy["schema_version"] != 1:
        raise ValueError("unsupported policy schema version")
    if identifier(policy.get("name")) == "unknown":
        raise ValueError("policy name must be a stable identifier")
    repeat = policy.get("minimum_repeat_count")
    if type(repeat) is not int or repeat < 1:
        raise ValueError("minimum_repeat_count must be a positive integer")
    for flag in ("require_baseline", "require_usage", "require_clean_revision"):
        if not isinstance(policy.get(flag), bool):
            raise ValueError(f"{flag} must be a boolean")
    for name, value in policy.items():
        if name.startswith("maximum_") and (
            number(value) is None or (name.endswith("_ratio") and value < 1)
        ):
            raise ValueError(
                f"{name} must be a finite nonnegative number (ratios at least 1)"
            )
    if policy.get("maximum_pass_rate_drop", 0) > 1:
        raise ValueError("maximum_pass_rate_drop must be at most 1")
    cases = policy.get("cases")
    if not isinstance(cases, dict) or not cases:
        raise ValueError("policy requires case rules")
    known = catalog()
    for case_id, rule in cases.items():
        if case_id not in known:
            raise ValueError(f"policy references unknown case: {case_id}")
        if (
            not isinstance(rule, dict)
            or set(rule) - CASE_FIELDS
            or "minimum_pass_rate" not in rule
        ):
            raise ValueError(f"invalid case policy: {case_id}")
        for name, value in rule.items():
            if number(value) is None or (name.startswith("minimum_") and value > 1):
                raise ValueError(f"invalid {name} for {case_id}")
    return policy

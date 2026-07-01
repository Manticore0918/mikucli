from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class JsonAction:
    kind: str
    name: str
    arguments: dict[str, Any]


def parse_json_action(text: str) -> JsonAction | None:
    payload = _extract_json(text)
    if payload is None:
        return None

    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError:
        return None

    if not isinstance(parsed, dict):
        return None

    if parsed.get("final") is not None:
        return JsonAction(kind="final", name="final", arguments={"content": str(parsed["final"])})

    name = parsed.get("tool") or parsed.get("action")
    arguments = parsed.get("arguments") or parsed.get("input") or {}
    if not isinstance(name, str) or not isinstance(arguments, dict):
        return None

    return JsonAction(kind="tool", name=name, arguments=arguments)


def _extract_json(text: str) -> str | None:
    stripped = text.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        return stripped

    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        return match.group(1)
    return None

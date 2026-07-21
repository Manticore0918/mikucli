from __future__ import annotations

import json
from typing import Any


def json_text(value: Any) -> str:
    return json.dumps(json_compatible(value), sort_keys=True, ensure_ascii=False)


def json_object(value: Any) -> dict[str, Any]:
    try:
        decoded = json.loads(str(value))
    except (TypeError, ValueError):
        return {}
    return decoded if isinstance(decoded, dict) else {}


def json_compatible(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_compatible(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_compatible(item) for item in value]
    if isinstance(value, tuple):
        return [json_compatible(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)

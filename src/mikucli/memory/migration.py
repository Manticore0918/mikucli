from __future__ import annotations

import json
from pathlib import Path

from .models import LongTermMemoryRecord, utc_now


def read_legacy_records(path: Path) -> list[LongTermMemoryRecord]:
    """Validate the complete legacy file before importing any of its facts."""
    if not path.exists():
        return []
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeError) as exc:
        raise ValueError(f"invalid legacy memory JSON at {path}: {exc}") from exc
    if isinstance(raw, dict):
        raw = raw.get("memories", [])
    if not isinstance(raw, list):
        raise TypeError(f"legacy memory at {path} must contain a list of facts")
    records = []
    for index, item in enumerate(raw):
        if (
            not isinstance(item, dict)
            or not isinstance(item.get("content"), str)
            or not item["content"].strip()
        ):
            raise ValueError(f"invalid legacy memory record {index + 1} in {path}")
        timestamp = item.get("created_at", "")
        if not isinstance(timestamp, str):
            raise TypeError(f"invalid timestamp in legacy memory record {index + 1} in {path}")
        records.append(
            LongTermMemoryRecord(
                item["content"].strip(), timestamp.strip() or utc_now()
            )
        )
    return records

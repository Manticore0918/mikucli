from __future__ import annotations

import json
from pathlib import Path

from .models import LongTermMemoryRecord, LongTermMemorySaveResult, utc_now
from .utilities import dedupe_key


class LongTermMemory:
    """Persist deduplicated durable facts for a workspace."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.records: list[LongTermMemoryRecord] = []
        self._keys: set[str] = set()
        self._load()

    def save(self, content: str) -> LongTermMemorySaveResult:
        cleaned = content.strip()
        if not cleaned:
            raise ValueError("long-term memory content cannot be empty")
        key = dedupe_key(cleaned)
        existing = self._find_by_key(key)
        if existing is not None:
            return LongTermMemorySaveResult(record=existing, saved=False)
        record = LongTermMemoryRecord(content=cleaned, created_at=utc_now())
        self.records.append(record)
        self._keys.add(key)
        self._write()
        return LongTermMemorySaveResult(record=record, saved=True)

    def _load(self) -> None:
        if not self.path.exists():
            return
        raw = json.loads(self.path.read_text(encoding="utf-8"))
        if isinstance(raw, dict):
            raw_records = raw.get("memories") or []
        elif isinstance(raw, list):
            raw_records = raw
        else:
            raw_records = []
        for item in raw_records:
            if not isinstance(item, dict):
                continue
            content = str(item.get("content") or "").strip()
            created_at = str(item.get("created_at") or "").strip()
            if not content:
                continue
            key = dedupe_key(content)
            if key in self._keys:
                continue
            self.records.append(LongTermMemoryRecord(content=content, created_at=created_at or utc_now()))
            self._keys.add(key)

    def _write(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "memories": [
                {"content": record.content, "created_at": record.created_at}
                for record in self.records
            ]
        }
        self.path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _find_by_key(self, key: str) -> LongTermMemoryRecord | None:
        for record in self.records:
            if dedupe_key(record.content) == key:
                return record
        return None


def default_long_term_memory_path(workspace: Path) -> Path:
    return workspace / ".mikucli" / "long_term_memory.json"

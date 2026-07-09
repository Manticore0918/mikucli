from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class RunLog:
    session_id: str
    task_prompt: str
    model: str
    workspace: str
    started_at: str = field(default_factory=_now)
    metadata: dict[str, Any] = field(default_factory=dict)
    events: list[dict[str, Any]] = field(default_factory=list)
    changed_paths: list[str] = field(default_factory=list)
    final_answer: str = ""

    def add_event(self, event_type: str, **payload: Any) -> None:
        self.events.append({"type": event_type, "at": _now(), **payload})

    def add_changed_paths(self, paths: list[str]) -> None:
        for path in paths:
            if path not in self.changed_paths:
                self.changed_paths.append(path)


class RunLogWriter:
    def __init__(self, workspace: Path) -> None:
        self.root = workspace / ".mikucli" / "runs"
        self.root.mkdir(parents=True, exist_ok=True)

    def write(self, run_log: RunLog) -> Path:
        path = self.root / f"{run_log.session_id}.json"
        path.write_text(json.dumps(asdict(run_log), indent=2), encoding="utf-8")
        return path


def new_session_id() -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{timestamp}-{uuid.uuid4().hex[:8]}"

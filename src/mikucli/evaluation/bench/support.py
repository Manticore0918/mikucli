from __future__ import annotations

import os
import re
from pathlib import Path

from mikucli.observability.store import StoreMode
from mikucli.tools import ToolApprovalRequest

from .models import ApprovalRecord


def snapshot_files(root: Path) -> dict[str, str]:
    files: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(root).as_posix()
        if rel == ".mikucli" or rel.startswith(".mikucli/"):
            continue
        files[rel] = _hash_file(path)
    return files


def approval_recorder(workspace: Path, approvals: list[ApprovalRecord]):
    workspace = workspace.resolve()

    def confirm(request: ToolApprovalRequest) -> bool:
        approved = Path(request.workspace).resolve() == workspace and _approval_details_are_local(request.details)
        approvals.append(
            ApprovalRecord(
                tool_name=request.tool_name,
                risk_level=request.risk_level.value,
                summary=request.summary,
                details=request.details,
                approved=approved,
            )
        )
        return approved

    return confirm


def safe_case_path(case_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", case_id)


def observability_store_mode() -> StoreMode:
    mode = os.environ.get("MIKUCLI_OBS_STORE", "sqlite").strip().casefold()
    if mode in {"sqlite", "jsonl", "both"}:
        return mode  # type: ignore[return-value]
    return "sqlite"


def _approval_details_are_local(details: str) -> bool:
    lowered = details.casefold()
    blocked = ("..", "~", "$home", "%userprofile%", "/etc/", "/home/")
    return not any(token in lowered for token in blocked)


def _hash_file(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()

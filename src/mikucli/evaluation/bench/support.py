from __future__ import annotations

import os
import re
from pathlib import Path

from mikucli.observability.store import StoreMode
from mikucli.tools import ToolApprovalRequest

from .models import ApprovalRecord
from .sandbox import is_safe_benchmark_test_command


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
        approved = _approval_is_safe(workspace, request)
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


def _approval_is_safe(workspace: Path, request: ToolApprovalRequest) -> bool:
    if Path(request.workspace).resolve() != workspace:
        return False
    if request.tool_name == "write_file":
        # ToolRegistry resolves and validates this path before it builds the request.
        return True
    if request.tool_name == "run_shell":
        _, marker, command = request.details.partition("\ncommand: ")
        return bool(marker) and is_safe_benchmark_test_command(command)
    # Benchmark fixtures do not need sensitive reads or high-risk MCP tools.
    return False


def _hash_file(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()

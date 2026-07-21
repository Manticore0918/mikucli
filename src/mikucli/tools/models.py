from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable


@dataclass(frozen=True)
class ToolResult:
    ok: bool
    content: str
    changed_paths: list[str] = field(default_factory=list)
    diff: str = ""


class ToolError(ValueError):
    pass


class ToolRiskLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass(frozen=True)
class ToolApprovalRequest:
    tool_name: str
    risk_level: ToolRiskLevel
    workspace: str
    summary: str
    details: str = ""


ConfirmTool = Callable[[ToolApprovalRequest], bool]

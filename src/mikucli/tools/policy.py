from __future__ import annotations

from .models import ToolError, ToolRiskLevel


class ToolPolicy:
    """Store the static risk classification for each built-in tool."""

    def __init__(self, risk_levels: dict[str, ToolRiskLevel] | None = None) -> None:
        self.risk_levels = risk_levels or {
            "list_files": ToolRiskLevel.LOW,
            "read_file": ToolRiskLevel.LOW,
            "write_file": ToolRiskLevel.MEDIUM,
            "run_shell": ToolRiskLevel.HIGH,
            "save_long_term_memory": ToolRiskLevel.LOW,
            "search_codebase": ToolRiskLevel.LOW,
        }

    def risk_for(self, tool_name: str) -> ToolRiskLevel:
        try:
            return self.risk_levels[tool_name]
        except KeyError as exc:
            raise ToolError(f"missing tool risk policy for: {tool_name}") from exc

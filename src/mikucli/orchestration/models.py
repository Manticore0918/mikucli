from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


StepStatus = Literal["pending", "running", "passed", "failed", "skipped"]


@dataclass(frozen=True)
class SubAgentSpec:
    id: str
    role: str
    purpose: str


@dataclass
class ExecutionStep:
    id: str
    task: str
    title: str = ""
    depends_on: list[str] = field(default_factory=list)
    status: StepStatus = "pending"
    assigned_worker: str = ""
    attempts: int = 0
    result: str = ""
    review_summary: str = ""
    feedback: str = ""
    skipped_reason: str = ""


@dataclass(frozen=True)
class ReviewDecision:
    approved: bool
    summary: str = ""
    issues: list[str] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return self.approved

    @property
    def feedback(self) -> str:
        return self.problems

    @property
    def problems(self) -> str:
        parts: list[str] = []
        if self.issues:
            parts.append("Issues: " + "; ".join(self.issues))
        if self.suggestions:
            parts.append("Suggestions: " + "; ".join(self.suggestions))
        return "\n".join(parts)


DEFAULT_SUBAGENTS: tuple[SubAgentSpec, ...] = (
    SubAgentSpec(
        id="planner-1",
        role="planner",
        purpose="Break down the task, identify dependencies, and produce an execution plan.",
    ),
    SubAgentSpec(
        id="worker-1",
        role="worker",
        purpose="Execute implementation work and gather concrete workspace evidence.",
    ),
    SubAgentSpec(
        id="worker-2",
        role="worker",
        purpose="Execute implementation work and gather concrete workspace evidence.",
    ),
    SubAgentSpec(
        id="reviewer-1",
        role="reviewer",
        purpose="Review completed steps for defects, missed requirements, and verification gaps.",
    ),
)

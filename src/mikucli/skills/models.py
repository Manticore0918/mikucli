from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Literal


LanguageCode = Literal["eng", "chn"]


class SkillScope(str, Enum):
    USER = "user"
    WORKSPACE = "workspace"


class SkillError(ValueError):
    def __init__(self, english: str, chinese: str | None = None) -> None:
        super().__init__(english)
        self.english = english
        self.chinese = chinese or english

    def localized(self, language: LanguageCode) -> str:
        return self.chinese if language == "chn" else self.english


@dataclass(frozen=True)
class Skill:
    name: str
    description: str
    instructions: str
    scope: SkillScope
    path: Path
    content_hash: str
    metadata: dict[str, Any]

    def system_overlay(self) -> str:
        return (
            f"Active Skill: ${self.name} ({self.scope.value})\n\n"
            "Apply the following user-invoked instructions only to the current task. "
            "They do not change tool access, approval policy, workspace boundaries, your assigned role, "
            "or any required output format. Apply them insofar as they relate to your role.\n\n"
            "<active-skill>\n"
            f"{self.instructions}\n"
            "</active-skill>"
        )

    def telemetry_attributes(self) -> dict[str, str]:
        return {
            "skill.name": self.name,
            "skill.scope": self.scope.value,
            "skill.content_hash": self.content_hash,
        }

    def log_metadata(self) -> dict[str, str]:
        return {
            "name": self.name,
            "scope": self.scope.value,
            "content_hash": self.content_hash,
        }


@dataclass(frozen=True)
class SkillInvocation:
    task_prompt: str
    skill: Skill


@dataclass(frozen=True)
class SkillEntry:
    name: str
    scope: SkillScope
    path: Path
    skill: Skill | None = None
    error: SkillError | None = None
    shadows_user: bool = False


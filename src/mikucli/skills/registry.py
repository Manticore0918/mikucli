from __future__ import annotations

import hashlib
from difflib import get_close_matches
from pathlib import Path
from typing import Any

import yaml

from .invocation import parse_skill_prefix, valid_skill_name
from .models import Skill, SkillEntry, SkillError, SkillInvocation, SkillScope


MAX_SKILL_CHARACTERS = 16_000
MAX_DESCRIPTION_CHARACTERS = 500
MAX_SKILL_BYTES = MAX_SKILL_CHARACTERS * 4


class SkillRegistry:
    def __init__(self, workspace: Path, *, user_root: Path | None = None) -> None:
        self.workspace = workspace.resolve()
        self.user_root = user_root or Path.home() / ".mikucli" / "skills"
        self.user_allowed_base = self.user_root.resolve(strict=False) if user_root else Path.home().resolve()
        self.workspace_root = self.workspace / ".mikucli" / "skills"

    def resolve_prompt(self, prompt: str) -> SkillInvocation | None:
        parsed = parse_skill_prefix(prompt)
        if parsed is None:
            return None
        name, task_prompt = parsed
        if not task_prompt:
            raise SkillError(
                f"Skill ${name} requires a task prompt. Usage: ${name} <task>",
                f"Skill ${name} 需要任务提示。用法：${name} <任务>",
            )
        return SkillInvocation(task_prompt=task_prompt, skill=self.resolve(name))

    def resolve(self, name: str) -> Skill:
        if not valid_skill_name(name):
            raise SkillError(
                f"invalid Skill name {name!r}; use lowercase kebab-case starting with a letter",
                f"无效的 Skill 名称 {name!r}；请使用以小写字母开头的 kebab-case",
            )
        workspace_path = self.workspace_root / name
        user_path = self.user_root / name
        if workspace_path.exists() or workspace_path.is_symlink():
            return self._load(workspace_path, SkillScope.WORKSPACE)
        if user_path.exists() or user_path.is_symlink():
            return self._load(user_path, SkillScope.USER)
        names = [entry.name for entry in self.list_entries() if valid_skill_name(entry.name)]
        suggestions = get_close_matches(name, names, n=3, cutoff=0.5)
        suggestion_text = f" Did you mean: {', '.join(f'${item}' for item in suggestions)}?" if suggestions else ""
        chinese_suggestion = f" 你是否想输入：{', '.join(f'${item}' for item in suggestions)}？" if suggestions else ""
        raise SkillError(
            f"unknown Skill ${name}.{suggestion_text}",
            f"未知 Skill ${name}。{chinese_suggestion}",
        )

    def list_entries(self) -> list[SkillEntry]:
        user_names = self._child_names(self.user_root)
        workspace_names = self._child_names(self.workspace_root)
        entries: list[SkillEntry] = []
        for name in sorted(user_names | workspace_names):
            if name in workspace_names:
                scope = SkillScope.WORKSPACE
                path = self.workspace_root / name
                shadows_user = name in user_names
            else:
                scope = SkillScope.USER
                path = self.user_root / name
                shadows_user = False
            if not valid_skill_name(name):
                error = SkillError(
                    f"invalid Skill directory name {name!r}",
                    f"无效的 Skill 目录名 {name!r}",
                )
                entries.append(
                    SkillEntry(name=name, scope=scope, path=path, error=error, shadows_user=shadows_user)
                )
                continue
            try:
                skill = self._load(path, scope)
            except SkillError as exc:
                entries.append(
                    SkillEntry(name=name, scope=scope, path=path, error=exc, shadows_user=shadows_user)
                )
            else:
                entries.append(
                    SkillEntry(name=name, scope=scope, path=path, skill=skill, shadows_user=shadows_user)
                )
        return entries

    @staticmethod
    def _child_names(root: Path) -> set[str]:
        if not root.exists() or not root.is_dir():
            return set()
        try:
            return {child.name for child in root.iterdir()}
        except OSError as exc:
            raise SkillError(f"could not list Skill root {root}: {exc}") from exc

    def _load(self, skill_directory: Path, scope: SkillScope) -> Skill:
        root = self.workspace_root if scope is SkillScope.WORKSPACE else self.user_root
        allowed_base = self.workspace if scope is SkillScope.WORKSPACE else self.user_allowed_base
        try:
            root_real = root.resolve(strict=True)
        except OSError as exc:
            raise SkillError(f"could not resolve Skill root {root}: {exc}") from exc
        if not root_real.is_relative_to(allowed_base):
            raise SkillError(f"Skill root escapes its allowed location: {root}")

        skill_file = skill_directory / "SKILL.md"
        try:
            file_real = skill_file.resolve(strict=True)
        except OSError as exc:
            raise SkillError(f"Skill ${skill_directory.name} is missing a readable SKILL.md: {exc}") from exc
        if not file_real.is_relative_to(root_real):
            raise SkillError(f"Skill ${skill_directory.name} resolves outside {root}")
        if not file_real.is_file():
            raise SkillError(f"Skill ${skill_directory.name} SKILL.md is not a regular file")
        try:
            if file_real.stat().st_size > MAX_SKILL_BYTES:
                raise SkillError(
                    f"Skill ${skill_directory.name} exceeds the {MAX_SKILL_CHARACTERS:,}-character limit"
                )
            raw_bytes = file_real.read_bytes()
            text = raw_bytes.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise SkillError(f"Skill ${skill_directory.name} SKILL.md must be UTF-8") from exc
        except OSError as exc:
            raise SkillError(f"could not read Skill ${skill_directory.name}: {exc}") from exc
        if len(text) > MAX_SKILL_CHARACTERS:
            raise SkillError(f"Skill ${skill_directory.name} exceeds the {MAX_SKILL_CHARACTERS:,}-character limit")

        metadata, instructions = self._parse_document(text, skill_directory.name)
        return Skill(
            name=metadata["name"],
            description=metadata["description"],
            instructions=instructions,
            scope=scope,
            path=file_real,
            content_hash=hashlib.sha256(raw_bytes).hexdigest(),
            metadata=metadata,
        )

    @staticmethod
    def _parse_document(text: str, directory_name: str) -> tuple[dict[str, Any], str]:
        lines = text.splitlines()
        if not lines or lines[0].strip() != "---":
            raise SkillError(f"Skill ${directory_name} SKILL.md must start with YAML frontmatter")
        try:
            closing_index = next(index for index, line in enumerate(lines[1:], start=1) if line.strip() == "---")
        except StopIteration as exc:
            raise SkillError(f"Skill ${directory_name} SKILL.md has unclosed YAML frontmatter") from exc
        try:
            metadata = yaml.safe_load("\n".join(lines[1:closing_index])) or {}
        except yaml.YAMLError as exc:
            raise SkillError(f"Skill ${directory_name} has invalid YAML frontmatter: {exc}") from exc
        if not isinstance(metadata, dict):
            raise SkillError(f"Skill ${directory_name} frontmatter must be a mapping")
        name = metadata.get("name")
        description = metadata.get("description")
        if not isinstance(name, str) or not name.strip():
            raise SkillError(f"Skill ${directory_name} frontmatter requires a string name")
        if name != directory_name:
            raise SkillError(f"Skill metadata name {name!r} must match directory {directory_name!r}")
        if not valid_skill_name(name):
            raise SkillError(f"Skill metadata name {name!r} is not valid lowercase kebab-case")
        if not isinstance(description, str) or not description.strip():
            raise SkillError(f"Skill ${directory_name} frontmatter requires a string description")
        description = description.strip()
        if len(description) > MAX_DESCRIPTION_CHARACTERS:
            raise SkillError(
                f"Skill ${directory_name} description exceeds the {MAX_DESCRIPTION_CHARACTERS}-character limit"
            )
        instructions = "\n".join(lines[closing_index + 1 :]).strip()
        if not instructions:
            raise SkillError(f"Skill ${directory_name} instruction body must not be empty")
        normalized_metadata = dict(metadata)
        normalized_metadata["name"] = name
        normalized_metadata["description"] = description
        return normalized_metadata, instructions

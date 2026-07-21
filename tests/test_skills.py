from __future__ import annotations

import io
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from mikucli.cli import _resolve_task_prompt, handle_slash_command
from mikucli.console import TerminalConsole
from mikucli.skills import (
    MAX_SKILL_CHARACTERS,
    SkillError,
    SkillRegistry,
    SkillScope,
    parse_skill_prefix,
    valid_skill_name,
)


class _UnusedCodebaseService:
    pass


def _write_skill(
    root: Path,
    name: str,
    *,
    description: str = "Test Skill.",
    instructions: str = "Follow the test instructions.",
    metadata_name: str | None = None,
) -> Path:
    directory = root / name
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "SKILL.md"
    path.write_text(
        "---\n"
        f"name: {metadata_name or name}\n"
        f"description: {description}\n"
        "future-field: preserved\n"
        "---\n\n"
        f"{instructions}\n",
        encoding="utf-8",
    )
    return path


class SkillInvocationTests(unittest.TestCase):
    def test_prefix_parser_consumes_one_leading_skill(self) -> None:
        self.assertEqual(parse_skill_prefix("$review-api inspect auth"), ("review-api", "inspect auth"))
        self.assertEqual(parse_skill_prefix("$review-api $security inspect auth"), ("review-api", "$security inspect auth"))

    def test_dollar_amount_and_invalid_case_are_ordinary_prompts(self) -> None:
        self.assertIsNone(parse_skill_prefix("$100 budget for migration"))
        self.assertIsNone(parse_skill_prefix("$Review inspect auth"))

    def test_skill_names_require_portable_lowercase_kebab_case(self) -> None:
        self.assertTrue(valid_skill_name("python-311"))
        self.assertFalse(valid_skill_name("311-python"))
        self.assertFalse(valid_skill_name("review_api"))
        self.assertFalse(valid_skill_name("a" * 65))


class SkillRegistryTests(unittest.TestCase):
    def test_workspace_skill_overrides_user_skill_and_preserves_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            user_root = root / "user-skills"
            workspace.mkdir()
            _write_skill(user_root, "review", instructions="user instructions")
            workspace_file = _write_skill(
                workspace / ".mikucli" / "skills",
                "review",
                instructions="workspace instructions",
            )

            skill = SkillRegistry(workspace, user_root=user_root).resolve("review")

            self.assertEqual(skill.scope, SkillScope.WORKSPACE)
            self.assertEqual(skill.instructions, "workspace instructions")
            self.assertEqual(skill.metadata["future-field"], "preserved")
            self.assertEqual(skill.path, workspace_file.resolve())
            self.assertEqual(len(skill.content_hash), 64)

    def test_invalid_workspace_skill_shadows_valid_user_skill(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            user_root = root / "user-skills"
            workspace.mkdir()
            _write_skill(user_root, "review", instructions="user instructions")
            _write_skill(
                workspace / ".mikucli" / "skills",
                "review",
                metadata_name="wrong-name",
            )
            registry = SkillRegistry(workspace, user_root=user_root)

            with self.assertRaisesRegex(SkillError, "must match directory"):
                registry.resolve("review")
            entry = registry.list_entries()[0]
            self.assertTrue(entry.shadows_user)
            self.assertIsNotNone(entry.error)
            self.assertIsNone(entry.skill)

    def test_resolve_prompt_requires_task_and_suggests_close_name(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            user_root = root / "user-skills"
            workspace.mkdir()
            _write_skill(user_root, "review-api")
            registry = SkillRegistry(workspace, user_root=user_root)

            with self.assertRaisesRegex(SkillError, "requires a task prompt"):
                registry.resolve_prompt("$review-api")
            with self.assertRaisesRegex(SkillError, r"Did you mean: \$review-api"):
                registry.resolve_prompt("$reviwe-api inspect auth")

    def test_registry_reloads_skill_on_every_invocation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            user_root = root / "user-skills"
            workspace.mkdir()
            path = _write_skill(user_root, "review", instructions="first version")
            registry = SkillRegistry(workspace, user_root=user_root)
            first = registry.resolve("review")
            path.write_text(
                "---\nname: review\ndescription: Reloaded.\n---\n\nsecond version\n",
                encoding="utf-8",
            )

            second = registry.resolve("review")

            self.assertEqual(first.instructions, "first version")
            self.assertEqual(second.instructions, "second version")
            self.assertNotEqual(first.content_hash, second.content_hash)

    def test_rejects_mismatched_metadata_oversized_files_and_empty_bodies(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            user_root = root / "user-skills"
            workspace.mkdir()
            _write_skill(user_root, "wrong", metadata_name="other")
            _write_skill(user_root, "empty", instructions="   ")
            _write_skill(user_root, "large", instructions="x" * MAX_SKILL_CHARACTERS)
            registry = SkillRegistry(workspace, user_root=user_root)

            with self.assertRaisesRegex(SkillError, "must match directory"):
                registry.resolve("wrong")
            with self.assertRaisesRegex(SkillError, "must not be empty"):
                registry.resolve("empty")
            with self.assertRaisesRegex(SkillError, "character limit"):
                registry.resolve("large")

    def test_rejects_skill_symlink_that_escapes_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            user_root = root / "user-skills"
            outside = root / "outside"
            workspace.mkdir()
            outside.mkdir()
            _write_skill(outside, "escape")
            user_root.mkdir()
            try:
                os.symlink(outside / "escape", user_root / "escape", target_is_directory=True)
            except OSError as exc:
                self.skipTest(f"directory symlinks are unavailable: {exc}")

            with self.assertRaisesRegex(SkillError, "resolves outside"):
                SkillRegistry(workspace, user_root=user_root).resolve("escape")

    def test_resolve_task_prompt_returns_stripped_task_and_skill(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            user_root = root / "user-skills"
            workspace.mkdir()
            _write_skill(user_root, "review")
            registry = SkillRegistry(workspace, user_root=user_root)

            task_prompt, skill = _resolve_task_prompt("$review inspect auth", registry)
            ordinary_prompt, ordinary_skill = _resolve_task_prompt("$100 budget", registry)

            self.assertEqual(task_prompt, "inspect auth")
            self.assertEqual(skill.name if skill else None, "review")
            self.assertEqual(ordinary_prompt, "$100 budget")
            self.assertIsNone(ordinary_skill)

    def test_skills_slash_command_lists_effective_registry(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            user_root = root / "user-skills"
            workspace.mkdir()
            _write_skill(user_root, "review", description="Review user code.")
            _write_skill(
                workspace / ".mikucli" / "skills",
                "review",
                description="Review workspace code.",
            )
            output = io.StringIO()

            with redirect_stdout(output):
                handled = handle_slash_command(
                    "/skills",
                    _UnusedCodebaseService(),  # type: ignore[arg-type]
                    TerminalConsole(),
                    skill_registry=SkillRegistry(workspace, user_root=user_root),
                )

            self.assertTrue(handled)
            self.assertIn("$review", output.getvalue())
            self.assertIn("Review workspace code.", output.getvalue())
            self.assertIn("overrides user", output.getvalue())

    def test_skills_slash_command_localizes_source_and_override_labels(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            user_root = root / "user-skills"
            workspace.mkdir()
            _write_skill(user_root, "review")
            _write_skill(workspace / ".mikucli" / "skills", "review")
            output = io.StringIO()

            with redirect_stdout(output):
                handle_slash_command(
                    "/skills",
                    _UnusedCodebaseService(),  # type: ignore[arg-type]
                    TerminalConsole("chn"),
                    skill_registry=SkillRegistry(workspace, user_root=user_root),
                )

            self.assertIn("工作区", output.getvalue())
            self.assertIn("覆盖用户 Skill", output.getvalue())


if __name__ == "__main__":
    unittest.main()

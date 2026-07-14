from __future__ import annotations

import tempfile
import unittest
import json
import sys
from pathlib import Path
from unittest.mock import patch

from mikucli.codebase.index import CodebaseIndexError
from mikucli.codebase.types import SearchResult
from mikucli.memory import LongTermMemory
from mikucli.tools import (
    MAX_READ_CHARS,
    MAX_READ_LINES,
    ToolApprovalRequest,
    ToolRegistry,
    ToolRiskLevel,
)
from mikucli.workspace import Workspace


class FakeCodebaseService:
    def __init__(self, missing: bool = False) -> None:
        self.missing = missing

    def search(self, query: str, limit: int = 8) -> list[SearchResult]:
        if self.missing:
            raise CodebaseIndexError("No Codebase Index exists. Run /index first.")
        return [
            SearchResult(
                path="src/app.py",
                start_line=1,
                end_line=3,
                kind="function",
                symbol="run",
                content="def run():\n    return True\n",
                hybrid_score=0.1,
                semantic_score=0.9,
                lexical_score=-1.0,
            )
        ]


class ToolTests(unittest.TestCase):
    def test_schemas_do_not_expose_tool_policy(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tools = ToolRegistry(Workspace(Path(tmp)))

            descriptions = "\n".join(schema["function"]["description"] for schema in tools.schemas())

            self.assertNotIn("risk", descriptions.lower())
            self.assertNotIn("approval", descriptions.lower())
            self.assertNotIn("review", descriptions.lower())

    def test_requires_approval_follows_tool_policy(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tools = ToolRegistry(Workspace(Path(tmp)))

            self.assertFalse(tools.requires_approval("read_file"))
            self.assertFalse(tools.requires_approval("read_file", {"path": "README.md"}))
            self.assertTrue(tools.requires_approval("read_file", {"path": ".env"}))
            self.assertTrue(tools.requires_approval("write_file"))
            self.assertTrue(tools.requires_approval("run_shell"))

    def test_read_file_schema_exposes_optional_line_range(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tools = ToolRegistry(Workspace(Path(tmp)))

            schema = next(item for item in tools.schemas() if item["function"]["name"] == "read_file")
            parameters = schema["function"]["parameters"]

            self.assertEqual(parameters["required"], ["path"])
            self.assertIn("start_line", parameters["properties"])
            self.assertIn("end_line", parameters["properties"])

    def test_read_file_returns_small_file_unchanged(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "note.md").write_text("one\ntwo\n", encoding="utf-8")
            tools = ToolRegistry(Workspace(root))

            result = tools.invoke("read_file", {"path": "note.md"})

            self.assertTrue(result.ok)
            self.assertEqual(result.content, "one\ntwo\n")

    def test_read_file_denies_sensitive_path_before_reading_contents(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / ".env").write_text("API_KEY=super-secret\n", encoding="utf-8")
            approvals: list[ToolApprovalRequest] = []
            tools = ToolRegistry(
                Workspace(root),
                confirm_tool=lambda request: approvals.append(request) or False,
            )

            with patch.object(Path, "read_text", side_effect=AssertionError("contents were read")):
                result = tools.invoke("read_file", {"path": ".env"})

            self.assertFalse(result.ok)
            self.assertEqual(result.content, "sensitive file read denied by user.")
            self.assertEqual(len(approvals), 1)
            self.assertEqual(approvals[0].tool_name, "read_file")
            self.assertEqual(approvals[0].risk_level, ToolRiskLevel.MEDIUM)
            self.assertEqual(approvals[0].summary, "Read sensitive file: .env")
            self.assertIn("No file contents have been read yet", approvals[0].details)
            self.assertNotIn("super-secret", approvals[0].details)

    def test_read_file_returns_sensitive_contents_after_user_approval(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / ".env.local").write_text("API_KEY=approved\n", encoding="utf-8")
            approvals: list[ToolApprovalRequest] = []
            tools = ToolRegistry(
                Workspace(root),
                confirm_tool=lambda request: approvals.append(request) or True,
            )

            result = tools.invoke("read_file", {"path": ".env.local"})

            self.assertTrue(result.ok)
            self.assertEqual(result.content, "API_KEY=approved\n")
            self.assertEqual(len(approvals), 1)

    def test_read_file_does_not_prompt_for_env_template(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / ".env.example").write_text("API_KEY=\n", encoding="utf-8")
            approvals: list[ToolApprovalRequest] = []
            tools = ToolRegistry(
                Workspace(root),
                confirm_tool=lambda request: approvals.append(request) or False,
            )

            result = tools.invoke("read_file", {"path": ".env.example"})

            self.assertTrue(result.ok)
            self.assertEqual(approvals, [])

    def test_read_file_rejects_unbounded_large_file_and_guides_agent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            content = "".join(f"line {index}\n" for index in range(1, MAX_READ_LINES + 2))
            (root / "large.md").write_text(content, encoding="utf-8")
            tools = ToolRegistry(Workspace(root))

            result = tools.invoke("read_file", {"path": "large.md"})

            self.assertFalse(result.ok)
            self.assertIn("too large for an unbounded read", result.content)
            self.assertIn("search_codebase", result.content)
            self.assertIn("start_line and end_line", result.content)

    def test_read_file_returns_llm_selected_inclusive_range(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "note.md").write_text("one\ntwo\nthree\nfour\n", encoding="utf-8")
            tools = ToolRegistry(Workspace(root))

            result = tools.invoke("read_file", {"path": "note.md", "start_line": 2, "end_line": 3})

            self.assertTrue(result.ok)
            self.assertEqual(result.content, "File: note.md\nLines: 2-3 of 4\n---\ntwo\nthree\n")

    def test_read_file_start_line_without_end_line_reads_bounded_range(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            content = "".join(f"{index}\n" for index in range(1, MAX_READ_LINES + 21))
            (root / "large.md").write_text(content, encoding="utf-8")
            tools = ToolRegistry(Workspace(root))

            result = tools.invoke("read_file", {"path": "large.md", "start_line": 21})

            self.assertTrue(result.ok)
            self.assertIn(f"Lines: 21-{MAX_READ_LINES + 20} of {MAX_READ_LINES + 20}", result.content)

    def test_read_file_rejects_invalid_or_oversized_ranges(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "large.md").write_text("x\n" * (MAX_READ_LINES + 1), encoding="utf-8")
            (root / "wide.md").write_text("x" * (MAX_READ_CHARS + 1) + "\n", encoding="utf-8")
            tools = ToolRegistry(Workspace(root))

            reversed_range = tools.invoke(
                "read_file", {"path": "large.md", "start_line": 10, "end_line": 9}
            )
            oversized_lines = tools.invoke(
                "read_file", {"path": "large.md", "start_line": 1, "end_line": MAX_READ_LINES + 1}
            )
            oversized_chars = tools.invoke(
                "read_file", {"path": "wide.md", "start_line": 1, "end_line": 1}
            )

            self.assertFalse(reversed_range.ok)
            self.assertIn("greater than or equal", reversed_range.content)
            self.assertFalse(oversized_lines.ok)
            self.assertIn("Choose a smaller range", oversized_lines.content)
            self.assertFalse(oversized_chars.ok)
            self.assertIn("Choose a smaller range", oversized_chars.content)

    def test_write_file_applies_change_and_returns_diff(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            approvals: list[ToolApprovalRequest] = []
            tools = ToolRegistry(Workspace(Path(tmp)), confirm_tool=lambda request: approvals.append(request) or True)
            result = tools.invoke("write_file", {"path": "note.txt", "content": "hello\n"})
            self.assertTrue(result.ok)
            self.assertEqual(result.changed_paths, ["note.txt"])
            self.assertIn("+++ b/note.txt", result.diff)
            self.assertEqual(approvals[0].tool_name, "write_file")
            self.assertIn("+++ b/note.txt", approvals[0].details)
            self.assertEqual((Path(tmp) / "note.txt").read_text(encoding="utf-8"), "hello\n")

    def test_write_file_denied_before_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "note.txt"
            tools = ToolRegistry(Workspace(Path(tmp)), confirm_tool=lambda _: False)

            result = tools.invoke("write_file", {"path": "note.txt", "content": "hello\n"})

            self.assertFalse(result.ok)
            self.assertEqual(result.content, "file change denied by user.")
            self.assertIn("+++ b/note.txt", result.diff)
            self.assertFalse(path.exists())

    def test_run_shell_denied_before_execution(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tools = ToolRegistry(Workspace(Path(tmp)), confirm_tool=lambda _: False)
            result = tools.invoke("run_shell", {"command": "echo hello", "reason": "test"})
            self.assertFalse(result.ok)
            self.assertEqual(result.content, "command denied by user.")

    def test_run_shell_sets_workspace_src_on_pythonpath(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "src" / "demo").mkdir(parents=True)
            (root / "src" / "demo" / "__init__.py").write_text("VALUE = 'ok'\n", encoding="utf-8")
            tools = ToolRegistry(Workspace(root), confirm_tool=lambda _: True)

            result = tools.invoke(
                "run_shell",
                {
                    "command": f'"{sys.executable}" -c "import demo; print(demo.VALUE)"',
                    "reason": "verify pythonpath",
                },
            )

            self.assertTrue(result.ok, result.content)
            self.assertEqual(result.content.strip(), "ok")

    def test_run_shell_accepts_posix_style_leading_env_assignment(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tools = ToolRegistry(Workspace(Path(tmp)), confirm_tool=lambda _: True)

            result = tools.invoke(
                "run_shell",
                {
                    "command": f'EXAMPLE_VALUE=ok "{sys.executable}" -c "import os; print(os.environ.get(\'EXAMPLE_VALUE\'))"',
                    "reason": "verify env assignment compatibility",
                },
            )

            self.assertTrue(result.ok, result.content)
            self.assertEqual(result.content.strip(), "ok")

    def test_list_files_omits_internal_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "visible.txt").write_text("ok", encoding="utf-8")
            (root / ".mikucli" / "runs").mkdir(parents=True)
            (root / ".mikucli" / "runs" / "log.json").write_text("{}", encoding="utf-8")
            tools = ToolRegistry(Workspace(root))
            result = tools.list_files()
            self.assertIn("visible.txt", result.content)
            self.assertNotIn(".mikucli", result.content)

    def test_save_long_term_memory_deduplicates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = Workspace(root)
            memory = LongTermMemory(root / ".mikucli" / "long_term_memory.json")
            approvals: list[ToolApprovalRequest] = []
            tools = ToolRegistry(
                workspace,
                confirm_tool=lambda request: approvals.append(request) or False,
                long_term_memory=memory,
            )

            first = tools.invoke("save_long_term_memory", {"content": "User prefers concise answers."})
            second = tools.invoke("save_long_term_memory", {"content": " user   prefers CONCISE answers. "})

            self.assertTrue(first.ok)
            self.assertTrue(second.ok)
            self.assertEqual(approvals, [])
            self.assertIn("already exists", second.content)
            payload = json.loads((root / ".mikucli" / "long_term_memory.json").read_text(encoding="utf-8"))
            self.assertEqual(len(payload["memories"]), 1)

    def test_search_codebase_formats_results(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tools = ToolRegistry(
                Workspace(Path(tmp)),
                codebase_service=FakeCodebaseService(),
            )

            result = tools.search_codebase("where is run?", limit=1)

            self.assertTrue(result.ok)
            self.assertIn("src/app.py:1-3", result.content)
            self.assertIn("hybrid=", result.content)

    def test_search_codebase_tells_user_to_index_first(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tools = ToolRegistry(
                Workspace(Path(tmp)),
                codebase_service=FakeCodebaseService(missing=True),
            )

            result = tools.search_codebase("anything")

            self.assertFalse(result.ok)
            self.assertIn("Run /index first", result.content)

    def test_search_codebase_filters_sensitive_paths_from_existing_index(self) -> None:
        class SensitiveResultService:
            def search(self, query: str, limit: int = 8) -> list[SearchResult]:
                return [
                    SearchResult(
                        path="config/.env.local",
                        start_line=1,
                        end_line=1,
                        kind="text",
                        symbol="",
                        content="API_KEY=secret\n",
                        hybrid_score=0.9,
                    )
                ]

        with tempfile.TemporaryDirectory() as tmp:
            tools = ToolRegistry(
                Workspace(Path(tmp)),
                codebase_service=SensitiveResultService(),
            )

            result = tools.search_codebase("API key")

            self.assertTrue(result.ok)
            self.assertNotIn("secret", result.content)
            self.assertNotIn(".env.local", result.content)


if __name__ == "__main__":
    unittest.main()

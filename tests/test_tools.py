from __future__ import annotations

import tempfile
import unittest
import json
import sys
from pathlib import Path

from mikucli.codebase.index import CodebaseIndexError
from mikucli.codebase.types import SearchResult
from mikucli.memory import LongTermMemory
from mikucli.tools import ToolApprovalRequest, ToolRegistry
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


if __name__ == "__main__":
    unittest.main()

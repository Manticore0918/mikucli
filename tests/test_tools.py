from __future__ import annotations

import tempfile
import unittest
import json
from pathlib import Path

from mikucli.codebase.index import CodebaseIndexError
from mikucli.codebase.types import SearchResult
from mikucli.memory import LongTermMemory
from mikucli.tools import ToolRegistry
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
    def test_write_file_applies_change_and_returns_diff(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tools = ToolRegistry(Workspace(Path(tmp)), confirm_command=lambda *_: False)
            result = tools.write_file("note.txt", "hello\n")
            self.assertTrue(result.ok)
            self.assertEqual(result.changed_paths, ["note.txt"])
            self.assertIn("+++ b/note.txt", result.diff)
            self.assertEqual((Path(tmp) / "note.txt").read_text(encoding="utf-8"), "hello\n")

    def test_run_shell_denied_before_execution(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tools = ToolRegistry(Workspace(Path(tmp)), confirm_command=lambda *_: False)
            result = tools.run_shell("echo hello", "test")
            self.assertFalse(result.ok)
            self.assertEqual(result.content, "command denied by user.")

    def test_list_files_omits_internal_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "visible.txt").write_text("ok", encoding="utf-8")
            (root / ".mikucli" / "runs").mkdir(parents=True)
            (root / ".mikucli" / "runs" / "log.json").write_text("{}", encoding="utf-8")
            tools = ToolRegistry(Workspace(root), confirm_command=lambda *_: False)
            result = tools.list_files()
            self.assertIn("visible.txt", result.content)
            self.assertNotIn(".mikucli", result.content)

    def test_save_long_term_memory_deduplicates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = Workspace(root)
            memory = LongTermMemory(root / ".mikucli" / "long_term_memory.json")
            tools = ToolRegistry(
                workspace,
                confirm_command=lambda *_: False,
                long_term_memory=memory,
            )

            first = tools.save_long_term_memory("User prefers concise answers.")
            second = tools.save_long_term_memory(" user   prefers CONCISE answers. ")

            self.assertTrue(first.ok)
            self.assertTrue(second.ok)
            self.assertIn("already exists", second.content)
            payload = json.loads((root / ".mikucli" / "long_term_memory.json").read_text(encoding="utf-8"))
            self.assertEqual(len(payload["memories"]), 1)

    def test_search_codebase_formats_results(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tools = ToolRegistry(
                Workspace(Path(tmp)),
                confirm_command=lambda *_: False,
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
                confirm_command=lambda *_: False,
                codebase_service=FakeCodebaseService(missing=True),
            )

            result = tools.search_codebase("anything")

            self.assertFalse(result.ok)
            self.assertIn("Run /index first", result.content)


if __name__ == "__main__":
    unittest.main()

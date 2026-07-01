from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from mikucli.workspace import Workspace, WorkspaceError


class WorkspaceTests(unittest.TestCase):
    def test_resolves_relative_paths_inside_workspace(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Workspace(Path(tmp))
            self.assertEqual(workspace.resolve("a/b.txt"), Path(tmp).resolve() / "a" / "b.txt")

    def test_rejects_parent_traversal(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Workspace(Path(tmp))
            with self.assertRaises(WorkspaceError):
                workspace.resolve("../outside.txt")


if __name__ == "__main__":
    unittest.main()

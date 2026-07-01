from __future__ import annotations

import importlib.util
import sqlite3
import tempfile
import unittest
from pathlib import Path

from mikucli.codebase.chunking import chunk_file
from mikucli.codebase.embeddings import EmbeddingError
from mikucli.codebase.files import select_index_files
from mikucli.codebase.index import ActiveIndex, CodebaseIndexReader, CodebaseIndexWriter
from mikucli.codebase.service import CodebaseService
from mikucli.codebase.types import CodeChunk, IndexedFile


class FakeEmbeddingClient:
    model = "fake-embed"

    def __init__(self, fail: bool = False) -> None:
        self.fail = fail
        self.inputs: list[str] = []

    def embed(self, inputs: list[str]) -> list[list[float]]:
        if self.fail:
            raise EmbeddingError("fake embedding failure")
        self.inputs.extend(inputs)
        return [_fake_vector(text) for text in inputs]


class CodebaseFileSelectionTests(unittest.TestCase):
    def test_selects_included_text_files_and_skips_internal_or_generated_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / ".gitignore").write_text("tests/ignored.py\n", encoding="utf-8")
            (root / "README.md").write_text("docs", encoding="utf-8")
            (root / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
            (root / "src" / "pkg").mkdir(parents=True)
            (root / "src" / "pkg" / "main.py").write_text("print('ok')\n", encoding="utf-8")
            (root / "tests").mkdir()
            (root / "tests" / "test_main.py").write_text("def test_ok(): pass\n", encoding="utf-8")
            (root / "tests" / "ignored.py").write_text("ignored\n", encoding="utf-8")
            (root / ".env").write_text("SECRET=1\n", encoding="utf-8")
            (root / "target").mkdir()
            (root / "target" / "generated.java").write_text("class Generated {}\n", encoding="utf-8")

            selection = select_index_files(root)

            paths = {file.path for file in selection.files}
            self.assertIn("README.md", paths)
            self.assertIn("pyproject.toml", paths)
            self.assertIn("src/pkg/main.py", paths)
            self.assertIn("tests/test_main.py", paths)
            self.assertNotIn("tests/ignored.py", paths)
            self.assertTrue(any(skip.path == ".env" for skip in selection.skips))
            self.assertTrue(any(skip.path == "target/generated.java" for skip in selection.skips))


class CodebaseChunkingTests(unittest.TestCase):
    def test_non_code_files_use_two_thousand_character_line_chunks(self) -> None:
        content = ("a" * 900 + "\n") * 3

        chunks = chunk_file("README.md", content)

        self.assertEqual(len(chunks), 2)
        self.assertEqual(chunks[0].kind, "text")
        self.assertLessEqual(len(chunks[0].content), 2000)
        self.assertEqual(chunks[0].start_line, 1)
        self.assertEqual(chunks[1].start_line, 3)

    @unittest.skipUnless(
        importlib.util.find_spec("tree_sitter")
        and importlib.util.find_spec("tree_sitter_python")
        and importlib.util.find_spec("tree_sitter_java"),
        "tree-sitter parser packages are not installed",
    )
    def test_code_files_use_tree_sitter_structural_chunks(self) -> None:
        python_chunks = chunk_file("src/app.py", "class App:\n    def run(self):\n        return True\n")
        java_chunks = chunk_file("src/App.java", "class App { String run() { return \"ok\"; } }\n")

        self.assertTrue(any(chunk.kind == "class" and chunk.symbol == "App" for chunk in python_chunks))
        self.assertTrue(any(chunk.kind == "method" and chunk.symbol == "run" for chunk in java_chunks))


class CodebaseIndexTests(unittest.TestCase):
    def test_hybrid_search_uses_semantic_and_lexical_scores(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            active = ActiveIndex(Path(tmp) / ".mikucli" / "codebase_index")
            active.root.mkdir(parents=True)
            file = IndexedFile("README.md", Path(tmp) / "README.md", 10, 1, "file-hash")
            chunks = [
                _chunk("README.md", 1, 1, "text", "handles config values"),
                _chunk("README.md", 2, 2, "text", "database migration notes"),
            ]
            embeddings = [_fake_vector(chunk.content) for chunk in chunks]
            with CodebaseIndexWriter(active, "fake-embed") as writer:
                writer.add_file(file, chunks, embeddings)
                writer.commit_active()

            results = CodebaseIndexReader(active).search(
                "handles config",
                _fake_vector("handles config"),
                limit=2,
            )

            self.assertEqual(results[0].content, "handles config values")
            self.assertGreater(results[0].hybrid_score, 0)
            self.assertIsNotNone(results[0].semantic_score)
            self.assertIsNotNone(results[0].lexical_score)

    def test_search_fails_when_index_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            active = ActiveIndex(Path(tmp) / ".mikucli" / "codebase_index")

            with self.assertRaisesRegex(Exception, "Run /index first"):
                CodebaseIndexReader(active).search("anything", [1.0], limit=1)

    def test_rebuild_keeps_previous_active_index_when_embedding_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "README.md").write_text("first index", encoding="utf-8")
            service = CodebaseService(
                workspace=root,
                embedding_provider="ollama",
                embedding_model="fake-embed",
                ollama_base_url="http://localhost:11434",
                embedding_client=FakeEmbeddingClient(),
            )
            service.rebuild_index()
            first_count = _indexed_file_count(service.index_path)

            (root / "pyproject.toml").write_text("[project]\nname='x'\n", encoding="utf-8")
            failing = CodebaseService(
                workspace=root,
                embedding_provider="ollama",
                embedding_model="fake-embed",
                ollama_base_url="http://localhost:11434",
                embedding_client=FakeEmbeddingClient(fail=True),
            )

            with self.assertRaises(EmbeddingError):
                failing.rebuild_index()

            self.assertEqual(_indexed_file_count(service.index_path), first_count)
            tmp_dir = root / ".mikucli" / "codebase_index" / "tmp"
            self.assertFalse(any(tmp_dir.glob("*.sqlite3")) if tmp_dir.exists() else False)


def _chunk(path: str, start_line: int, end_line: int, kind: str, content: str) -> CodeChunk:
    return CodeChunk(
        path=path,
        start_line=start_line,
        end_line=end_line,
        kind=kind,
        symbol="",
        content=content,
        content_hash=content,
        chunk_hash=f"{path}:{start_line}:{end_line}:{content}",
    )


def _fake_vector(text: str) -> list[float]:
    return [
        float("handles" in text.casefold()),
        float("config" in text.casefold()),
        float("database" in text.casefold()),
    ]


def _indexed_file_count(path: Path) -> int:
    conn = sqlite3.connect(path)
    try:
        return int(conn.execute("SELECT COUNT(*) FROM files").fetchone()[0])
    finally:
        conn.close()


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import Mock, patch

from mikucli.cli import main as agent_main
from mikucli.llm import AssistantMessage, TokenUsage
from mikucli.memory import (
    LongTermMemory,
    MapReduceContextCompressor,
    MemoryEntry,
    MemoryStorageError,
    MemoryType,
    SessionMemory,
    default_long_term_memory_path,
)
from mikucli.tools import ToolRegistry
from mikucli.workspace import Workspace


class MemoryIntegrationTests(unittest.TestCase):
    def test_failed_fact_persistence_preserves_compression_sources_for_retry(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            memory = LongTermMemory(
                default_long_term_memory_path(Path(directory)),
                warning_callback=lambda _: None,
            )
            session = SessionMemory(
                {"role": "system", "content": "system"}, long_term_memory=memory
            )
            entry = MemoryEntry(
                type=MemoryType.CONVERSATION,
                messages=[],
                content="important original context",
            )
            session.old_entries.append(entry)
            responses = [
                AssistantMessage(
                    content=text, tool_calls=[], raw={}, token_usage=TokenUsage()
                )
                for text in (
                    "summary",
                    '["durable fact"]',
                    "summary",
                    '["durable fact"]',
                )
            ]
            client = Mock()
            client.chat.side_effect = responses
            compressor = MapReduceContextCompressor(
                client=client, model="test", long_term_memory=memory
            )
            with (
                patch.object(
                    memory._store, "save_many", side_effect=PermissionError("read only")
                ),
                self.assertRaises(MemoryStorageError),
            ):
                session.compress_old_entries(compressor)
            self.assertEqual(session.old_entries, [entry])
            self.assertEqual(session.summary_entries, [])
            summary = session.compress_old_entries(compressor)
            self.assertEqual(summary.metadata["saved_fact_count"], 1)
            self.assertEqual(session.old_entries, [])
            self.assertEqual(
                [record.content for record in memory.records], ["durable fact"]
            )

    def test_tool_reports_unavailable_memory_as_failure(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = default_long_term_memory_path(root)
            path.parent.mkdir()
            path.write_bytes(b"broken database")
            memory = LongTermMemory(path, warning_callback=lambda _: None)
            result = ToolRegistry(Workspace(root), long_term_memory=memory).invoke(
                "save_long_term_memory", {"content": "fact"}
            )
            self.assertFalse(result.ok)
            self.assertIn("Persistent memory unavailable", result.content)
            self.assertEqual(result.changed_paths, [])

    def test_agent_can_answer_when_persistent_memory_is_corrupt(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = default_long_term_memory_path(root)
            path.parent.mkdir()
            path.write_bytes(b"broken database")
            output, errors = StringIO(), StringIO()
            client = Mock()
            client.chat.return_value = AssistantMessage(
                content="still usable", tool_calls=[], raw={}, token_usage=TokenUsage()
            )
            with (
                patch("mikucli.cli.ModelClient", return_value=client),
                redirect_stdout(output),
                redirect_stderr(errors),
            ):
                code = agent_main(
                    ["--workspace", str(root), "--api-key", "test-only", "hello"]
                )
            self.assertEqual(code, 0)
            self.assertIn("still usable", output.getvalue())
            self.assertIn("Persistent memory unavailable", errors.getvalue())
            self.assertEqual(path.read_bytes(), b"broken database")

    def test_backup_and_restore_commands_require_no_api_key(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = default_long_term_memory_path(root)
            memory = LongTermMemory(path)
            saved = memory.save("valuable fact")
            snapshot = root / "manual.bak"
            environment = {
                key: value
                for key, value in os.environ.items()
                if not key.startswith(("MIKUCLI_", "BIGMODEL_"))
            }
            environment["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")

            def command(*args: str) -> subprocess.CompletedProcess[str]:
                return subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "mikucli.memory",
                        *args,
                        "--workspace",
                        str(root),
                    ],
                    capture_output=True,
                    text=True,
                    env=environment,
                    timeout=15,
                    check=False,
                )

            backed_up = command("backup", "--output", str(snapshot))
            self.assertEqual(backed_up.returncode, 0, backed_up.stderr)
            path.write_bytes(b"broken database")
            restored = command("restore", "--backup", str(snapshot))
            self.assertEqual(restored.returncode, 0, restored.stderr)
            self.assertIn("Previous files preserved", restored.stdout)
            self.assertEqual(LongTermMemory(path).records, [saved.record])
            snapshot.write_bytes(b"invalid backup")
            refused = command("restore", "--backup", str(snapshot))
            self.assertEqual(refused.returncode, 1)
            self.assertIn("original files are preserved", refused.stderr)
            self.assertEqual(LongTermMemory(path).records, [saved.record])


if __name__ == "__main__":
    unittest.main()

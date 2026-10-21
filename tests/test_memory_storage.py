from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from pathlib import Path
from unittest.mock import patch

from mikucli.memory import (
    LongTermMemory,
    MemoryStorageError,
    default_long_term_memory_path,
)
from mikucli.memory.recovery import restore_memory
from mikucli.memory.storage import restore_marker


class DurableMemoryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.path = default_long_term_memory_path(Path(self.temporary.name))
        self.messages: list[str] = []

    def memory(self, **kwargs) -> LongTermMemory:
        return LongTermMemory(
            self.path, warning_callback=self.messages.append, **kwargs
        )

    def test_independent_instances_read_each_others_committed_facts(self) -> None:
        first, second = self.memory(), self.memory()
        a = first.save("first fact")
        b = second.save("second fact")
        self.assertEqual(first.records, [a.record, b.record])
        self.assertEqual(second.records, first.records)
        self.assertEqual(self.memory().records, first.records)

    def test_same_instance_can_be_used_by_worker_threads(self) -> None:
        memory = self.memory()
        with ThreadPoolExecutor(max_workers=4) as executor:
            results = list(
                executor.map(
                    memory.save,
                    ["same fact", *[f"fact {i}" for i in range(8)], "SAME FACT"],
                )
            )
        self.assertEqual(sum(result.saved for result in results), 9)
        self.assertEqual(len(self.memory().records), 9)
        self.assertEqual(self.messages, [])

    def test_failed_commit_is_not_cached_and_can_be_retried(self) -> None:
        memory = self.memory()
        existing = memory.save("existing fact")
        real_connect = sqlite3.connect

        class FailedCommit(sqlite3.Connection):
            def commit(self) -> None:
                raise sqlite3.OperationalError("simulated disk full")

        with (
            patch(
                "mikucli.memory.storage.sqlite3.connect",
                side_effect=lambda *a, **kw: real_connect(
                    *a, **kw, factory=FailedCommit
                ),
            ),
            self.assertRaisesRegex(MemoryStorageError, "disk full"),
        ):
            memory.save("new fact")
        self.assertEqual(memory.records, [existing.record])
        retried = memory.save("new fact")
        self.assertTrue(retried.saved)
        self.assertEqual(self.memory().records, [existing.record, retried.record])

    def test_fact_batch_rolls_back_when_one_insert_fails(self) -> None:
        memory = self.memory()
        with closing(sqlite3.connect(self.path)) as connection:
            connection.execute(
                "CREATE TRIGGER reject_fact BEFORE INSERT ON memories WHEN NEW.content='reject' "
                "BEGIN SELECT RAISE(ABORT, 'rejected write'); END"
            )
            connection.commit()
        with self.assertRaisesRegex(MemoryStorageError, "rejected write"):
            memory.save_many(["first fact", "reject"])
        self.assertEqual(memory.records, [])
        with closing(sqlite3.connect(self.path)) as connection:
            connection.execute("DROP TRIGGER reject_fact")
            connection.commit()
        self.assertTrue(
            all(result.saved for result in memory.save_many(["first fact", "reject"]))
        )

    def test_busy_writer_has_bounded_failure_then_successful_retry(self) -> None:
        memory = self.memory(timeout=0.05)
        with closing(sqlite3.connect(self.path, isolation_level=None)) as writer:
            writer.execute("BEGIN IMMEDIATE")
            with self.assertRaisesRegex(MemoryStorageError, "locked"):
                memory.save("retry later")
            writer.rollback()
        self.assertTrue(memory.save("retry later").saved)

    def test_legacy_json_is_preserved_and_imported_once(self) -> None:
        legacy = self.path.with_suffix(".json")
        legacy.parent.mkdir()
        original = json.dumps(
            {
                "memories": [
                    {
                        "content": "First fact",
                        "created_at": "2026-01-01T00:00:00+00:00",
                    },
                    {
                        "content": " FIRST   FACT ",
                        "created_at": "2026-02-01T00:00:00+00:00",
                    },
                    {"content": "Second fact"},
                ]
            }
        )
        legacy.write_text(original, encoding="utf-8")
        memory = LongTermMemory(legacy, warning_callback=self.messages.append)
        result = memory.save("first FACT")
        self.assertFalse(result.saved)
        self.assertEqual(result.record.created_at, "2026-01-01T00:00:00+00:00")
        self.assertEqual(legacy.read_text(encoding="utf-8"), original)
        legacy.write_text("broken old file", encoding="utf-8")
        self.assertEqual(len(self.memory().records), 2)
        self.assertEqual(self.messages, [])

    def test_legacy_list_format_is_supported(self) -> None:
        legacy = self.path.with_suffix(".json")
        legacy.parent.mkdir()
        legacy.write_text('[{"content": "legacy fact"}]', encoding="utf-8")
        self.assertEqual(
            [record.content for record in self.memory().records], ["legacy fact"]
        )

    def test_bad_legacy_data_is_never_silently_replaced(self) -> None:
        legacy = self.path.with_suffix(".json")
        legacy.parent.mkdir()
        for content in (
            '{"memories": [',
            '{"memories": [{"content": 7}]}',
            '{"memories": "wrong"}',
        ):
            with self.subTest(content=content):
                legacy.write_text(content, encoding="utf-8")
                memory = self.memory()
                self.assertFalse(memory.available)
                self.assertIn(str(legacy), memory.error)
                self.assertEqual(memory.records, [])
                with self.assertRaises(MemoryStorageError):
                    memory.save("new fact")
                self.assertEqual(legacy.read_text(encoding="utf-8"), content)
        legacy.write_text('[{"content":"repaired"}]', encoding="utf-8")
        self.assertTrue(memory.save("new fact").saved)
        self.assertEqual(
            [record.content for record in memory.records], ["repaired", "new fact"]
        )

    def test_corrupt_database_is_preserved_and_restored_from_verified_backup(
        self,
    ) -> None:
        memory = self.memory()
        saved = memory.save("valuable fact")
        memory.path.write_bytes(b"damaged database")
        broken = self.memory()
        self.assertFalse(broken.available)
        self.assertEqual(broken.records, [])
        with self.assertRaises(MemoryStorageError):
            broken.save("must not succeed")
        self.assertEqual(memory.path.read_bytes(), b"damaged database")
        archive = restore_memory(memory.path)
        self.assertEqual((archive / memory.path.name).read_bytes(), b"damaged database")
        self.assertEqual(self.memory().records, [saved.record])
        self.assertTrue(memory.backup_path.exists())

    def test_bad_backup_cannot_replace_the_primary(self) -> None:
        memory = self.memory()
        saved = memory.save("valuable fact")
        backup = self.path.parent / "bad.bak"
        backup.write_bytes(b"not a snapshot")
        with self.assertRaises(MemoryStorageError):
            restore_memory(self.path, backup)
        self.assertEqual(self.memory().records, [saved.record])
        self.assertFalse(restore_marker(self.path).exists())

    def test_backup_failure_does_not_misreport_a_committed_save(self) -> None:
        memory = self.memory()
        original_backup = memory.backup_path.read_bytes()
        with patch.object(
            memory._store, "backup", side_effect=PermissionError("backup read only")
        ):
            saved = memory.save("durable primary")
        self.assertTrue(saved.saved)
        self.assertEqual(self.memory().records, [saved.record])
        self.assertEqual(memory.backup_path.read_bytes(), original_backup)
        self.assertTrue(
            any("backup could not be updated" in message for message in self.messages)
        )

    def test_backup_cannot_overwrite_storage_files(self) -> None:
        memory = self.memory()
        for destination in (
            self.path,
            self.path.with_suffix(".json"),
            restore_marker(self.path),
            self.path.with_name(self.path.name + "-journal"),
        ):
            with (
                self.subTest(destination=destination),
                self.assertRaises(MemoryStorageError),
            ):
                memory.backup(destination)
        self.assertTrue(memory.save("still works").saved)

    def test_interrupted_restore_blocks_startup_until_explicit_resume(self) -> None:
        memory = self.memory()
        saved = memory.save("valuable fact")
        with (
            patch(
                "mikucli.memory.recovery.os.replace",
                side_effect=OSError("interrupted replacement"),
            ),
            self.assertRaises(MemoryStorageError),
        ):
            restore_memory(self.path)
        self.assertTrue(restore_marker(self.path).exists())
        self.assertFalse(self.memory().available)
        with self.assertRaisesRegex(MemoryStorageError, "--resume"):
            restore_memory(self.path)
        restore_memory(self.path, resume=True)
        self.assertFalse(restore_marker(self.path).exists())
        self.assertEqual(self.memory().records, [saved.record])

    def test_missing_database_does_not_silently_create_empty_memory_over_a_backup(
        self,
    ) -> None:
        memory = self.memory()
        saved = memory.save("valuable fact")
        self.path.unlink()
        self.assertFalse(self.memory().available)
        self.assertFalse(self.path.exists())
        restore_memory(self.path)
        self.assertEqual(self.memory().records, [saved.record])

    def test_future_schema_is_rejected_without_rewriting_it(self) -> None:
        self.memory()
        with closing(sqlite3.connect(self.path)) as connection:
            connection.execute("PRAGMA user_version=99")
        original = self.path.read_bytes()
        self.assertFalse(self.memory().available)
        self.assertEqual(self.path.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()

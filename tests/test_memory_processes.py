from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

from mikucli.memory import LongTermMemory, default_long_term_memory_path


class MemoryProcessTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.path = default_long_term_memory_path(self.root)
        self.environment = os.environ.copy()
        self.environment["PYTHONPATH"] = str(
            Path(__file__).resolve().parents[1] / "src"
        )

    def child(self, code: str, *arguments: str) -> subprocess.Popen[str]:
        process = subprocess.Popen(
            [sys.executable, "-c", code, str(self.path), *arguments],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=self.environment,
        )

        def cleanup() -> None:
            if process.poll() is None:
                process.kill()
            process.communicate(timeout=10)

        self.addCleanup(cleanup)
        return process

    def wait_ready(self, process: subprocess.Popen[str], signal: Path) -> None:
        deadline = time.monotonic() + 10
        while not signal.exists():
            if process.poll() is not None:
                self.fail(
                    f"child exited before synchronization: {process.communicate()}"
                )
            if time.monotonic() >= deadline:
                self.fail("child did not reach synchronization point")
            time.sleep(0.01)

    def test_concurrent_processes_preserve_unique_facts_and_deduplicate_shared_fact(
        self,
    ) -> None:
        # Start from legacy JSON so these processes also race on first migration.
        legacy = self.path.with_suffix(".json")
        legacy.parent.mkdir()
        legacy.write_text(
            '[{"content":"legacy fact","created_at":"2026-01-01"}]', encoding="utf-8"
        )
        gate = self.root / "go"
        code = """
import json, sys, time
from pathlib import Path
from mikucli.memory import LongTermMemory
path, gate, ready, number = map(Path, sys.argv[1:])
memory = LongTermMemory(path)
ready.touch()
while not gate.exists():
    time.sleep(0.01)
results = memory.save_many(['shared fact', f'unique {number}'])
print(json.dumps([{'saved': r.saved, 'created_at': r.record.created_at} for r in results]))
"""
        processes = []
        for index in range(4):
            ready = self.root / f"ready-{index}"
            process = self.child(code, str(gate), str(ready), str(index))
            processes.append(process)
            self.wait_ready(process, ready)
        gate.touch()
        shared = []
        for process in processes:
            output, errors = process.communicate(timeout=20)
            self.assertEqual(process.returncode, 0, errors)
            self.assertEqual(errors, "")
            shared.append(json.loads(output)[0])
        self.assertEqual(sum(item["saved"] for item in shared), 1)
        self.assertEqual(len({item["created_at"] for item in shared}), 1)
        memory = LongTermMemory(self.path)
        self.assertEqual(
            {record.content for record in memory.records},
            {"legacy fact", "shared fact", *(f"unique {i}" for i in range(4))},
        )
        self.assertEqual(
            legacy.read_text(encoding="utf-8"),
            '[{"content":"legacy fact","created_at":"2026-01-01"}]',
        )
        self.assertEqual(LongTermMemory(memory.backup_path).records, memory.records)

    def test_killed_writer_rolls_back_its_uncommitted_fact(self) -> None:
        memory = LongTermMemory(self.path)
        saved = memory.save("already committed")
        ready = self.root / "ready"
        process = self.child(
            """
import sys
from pathlib import Path
from mikucli.memory import LongTermMemory
memory = LongTermMemory(Path(sys.argv[1]))
with memory._store.connect() as connection:
    connection.execute('BEGIN IMMEDIATE')
    connection.execute('INSERT INTO memories VALUES (?, ?, ?)', ('uncommitted', 'uncommitted', '2026-01-01'))
    Path(sys.argv[2]).touch()
    sys.stdin.read(1)
""",
            str(ready),
        )
        self.wait_ready(process, ready)
        process.kill()
        process.communicate(timeout=10)
        reopened = LongTermMemory(self.path)
        self.assertTrue(reopened.available)
        self.assertEqual(reopened.records, [saved.record])
        self.assertTrue(reopened.save("next committed fact").saved)

    def test_killed_migration_rolls_back_and_reimports_on_next_startup(self) -> None:
        legacy = self.path.with_suffix(".json")
        legacy.parent.mkdir()
        original = '[{"content":"first","created_at":"2026-01-01"},{"content":"second","created_at":"2026-01-02"}]'
        legacy.write_text(original, encoding="utf-8")
        ready = self.root / "ready"
        process = self.child(
            """
import sys
from pathlib import Path
from mikucli.memory import LongTermMemory
from mikucli.memory import storage
original_reader = storage.read_legacy_records
def interrupted_reader(path):
    records = original_reader(path)
    yield records[0]
    Path(sys.argv[2]).touch()
    sys.stdin.read(1)
    yield from records[1:]
storage.read_legacy_records = interrupted_reader
LongTermMemory(Path(sys.argv[1]))
""",
            str(ready),
        )
        self.wait_ready(process, ready)
        process.kill()
        process.communicate(timeout=10)
        reopened = LongTermMemory(self.path)
        self.assertTrue(reopened.available)
        self.assertEqual(
            [record.content for record in reopened.records], ["first", "second"]
        )
        self.assertEqual(legacy.read_text(encoding="utf-8"), original)
        self.assertEqual(LongTermMemory(self.path).records, reopened.records)


if __name__ == "__main__":
    unittest.main()

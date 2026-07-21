from __future__ import annotations

import unittest
from pathlib import Path


SOURCE_ROOT = Path(__file__).parents[1] / "src" / "mikucli"
MAX_SOURCE_LINES = 400


class SourceStructureTests(unittest.TestCase):
    def test_production_modules_stay_below_line_budget(self) -> None:
        oversized: list[str] = []
        for path in sorted(SOURCE_ROOT.rglob("*.py")):
            line_count = len(path.read_text(encoding="utf-8").splitlines())
            if line_count > MAX_SOURCE_LINES:
                oversized.append(f"{path.relative_to(SOURCE_ROOT)}: {line_count} lines")

        self.assertEqual(
            oversized,
            [],
            "Split oversized modules by responsibility:\n" + "\n".join(oversized),
        )


if __name__ == "__main__":
    unittest.main()

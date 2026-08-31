from __future__ import annotations

import unittest
import warnings
from pathlib import Path


SOURCE_ROOT = Path(__file__).parents[1] / "src" / "mikucli"
WARN_SOURCE_LINES = 400
MAX_SOURCE_LINES = 600


class SourceStructureTests(unittest.TestCase):
    def test_production_modules_stay_below_line_budget(self) -> None:
        oversized: list[str] = []
        for path in sorted(SOURCE_ROOT.rglob("*.py")):
            line_count = len(path.read_text(encoding="utf-8").splitlines())
            if line_count > MAX_SOURCE_LINES:
                oversized.append(f"{path.relative_to(SOURCE_ROOT)}: {line_count} lines")
            elif line_count > WARN_SOURCE_LINES:
                warnings.warn(
                    f"Consider splitting {path.relative_to(SOURCE_ROOT)}: "
                    f"{line_count} lines exceeds the {WARN_SOURCE_LINES}-line target",
                    UserWarning,
                    stacklevel=2,
                )

        self.assertEqual(
            oversized,
            [],
            f"Split modules exceeding the {MAX_SOURCE_LINES}-line hard limit "
            "by responsibility:\n" + "\n".join(oversized),
        )


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import unittest

from showcase.team_review.inventory import allocate_units


class AllocateUnitsTests(unittest.TestCase):
    def test_allocates_available_units(self) -> None:
        self.assertEqual(allocate_units(available_units=12, requested_units=5), 7)

    def test_rejects_request_above_inventory(self) -> None:
        with self.assertRaisesRegex(ValueError, "exceeds available inventory"):
            allocate_units(available_units=2, requested_units=3)

    def test_rejects_zero_unit_request(self) -> None:
        with self.assertRaisesRegex(ValueError, "positive integer"):
            allocate_units(available_units=12, requested_units=0)

    def test_rejects_negative_unit_request(self) -> None:
        with self.assertRaisesRegex(ValueError, "positive integer"):
            allocate_units(available_units=12, requested_units=-4)


if __name__ == "__main__":
    unittest.main()

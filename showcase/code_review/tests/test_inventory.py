from __future__ import annotations

import unittest

from showcase.code_review.inventory import ReservationRequest, remaining_units


class RemainingUnitsTests(unittest.TestCase):
    def test_reserves_available_stock(self) -> None:
        request = ReservationRequest(sku="MUG-01", available_units=8, requested_units=3)

        self.assertEqual(remaining_units(request), 5)

    def test_rejects_request_above_available_stock(self) -> None:
        request = ReservationRequest(sku="MUG-01", available_units=2, requested_units=3)

        with self.assertRaisesRegex(ValueError, "exceeds available stock"):
            remaining_units(request)

    def test_rejects_zero_unit_request(self) -> None:
        request = ReservationRequest(sku="MUG-01", available_units=8, requested_units=0)

        with self.assertRaisesRegex(ValueError, "at least one"):
            remaining_units(request)

    def test_rejects_negative_unit_request(self) -> None:
        request = ReservationRequest(sku="MUG-01", available_units=8, requested_units=-2)

        with self.assertRaisesRegex(ValueError, "at least one"):
            remaining_units(request)


if __name__ == "__main__":
    unittest.main()


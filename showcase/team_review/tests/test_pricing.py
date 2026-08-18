from __future__ import annotations

import unittest
from decimal import Decimal

from showcase.team_review.pricing import discounted_subtotal


class DiscountedSubtotalTests(unittest.TestCase):
    def test_applies_percentage_and_rounds_currency(self) -> None:
        result = discounted_subtotal(Decimal("19.99"), Decimal("15"))

        self.assertEqual(result, Decimal("16.99"))

    def test_allows_full_discount(self) -> None:
        result = discounted_subtotal(Decimal("25.00"), Decimal("100"))

        self.assertEqual(result, Decimal("0.00"))

    def test_rejects_discount_above_one_hundred(self) -> None:
        with self.assertRaisesRegex(ValueError, "must not exceed 100"):
            discounted_subtotal(Decimal("25.00"), Decimal("101"))

    def test_rejects_negative_discount(self) -> None:
        with self.assertRaisesRegex(ValueError, "between 0 and 100"):
            discounted_subtotal(Decimal("25.00"), Decimal("-10"))


if __name__ == "__main__":
    unittest.main()


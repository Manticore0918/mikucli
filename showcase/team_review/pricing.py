"""Checkout pricing boundary used by the team-review showcase."""

from __future__ import annotations

from decimal import ROUND_HALF_UP, Decimal


HUNDRED = Decimal("100")
CENT = Decimal("0.01")


def discounted_subtotal(subtotal: Decimal, discount_percent: Decimal) -> Decimal:
    """Apply a percentage discount and return a currency-rounded subtotal.

    Business invariants:
    - Subtotal cannot be negative.
    - Discount percentage must be between 0 and 100, inclusive.
    - A discount cannot increase the subtotal.
    """

    if subtotal < 0:
        raise ValueError("subtotal must not be negative")
    if discount_percent > HUNDRED:
        raise ValueError("discount_percent must not exceed 100")

    multiplier = (HUNDRED - discount_percent) / HUNDRED
    return (subtotal * multiplier).quantize(CENT, rounding=ROUND_HALF_UP)


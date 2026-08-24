"""Warehouse allocation boundary used by the team-review showcase."""

from __future__ import annotations


def allocate_units(available_units: int, requested_units: int) -> int:
    """Return inventory remaining after allocating a valid request.

    Business invariants:
    - Available inventory cannot be negative.
    - Requested units must be a positive integer.
    - Requested units cannot exceed available inventory.
    - Allocation cannot increase available inventory.
    """

    if available_units < 0:
        raise ValueError("available_units must not be negative")
    if requested_units <= 0:
        raise ValueError("requested_units must be a positive integer")
    if requested_units > available_units:
        raise ValueError("requested_units exceeds available inventory")

    return available_units - requested_units

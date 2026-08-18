"""Inventory reservation boundary used by the short review showcase."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ReservationRequest:
    """A request to reserve units from the current stock level."""

    sku: str
    available_units: int
    requested_units: int


def remaining_units(request: ReservationRequest) -> int:
    """Return stock remaining after a valid reservation.

    Business invariants:
    - SKU must not be blank.
    - Available stock cannot be negative.
    - A reservation must request at least one unit.
    - A reservation cannot exceed the available stock.
    - Successful reservations can only reduce stock.
    """

    if not request.sku.strip():
        raise ValueError("sku must not be blank")
    if request.available_units < 0:
        raise ValueError("available_units must not be negative")
    if request.requested_units > request.available_units:
        raise ValueError("requested_units exceeds available stock")

    return request.available_units - request.requested_units


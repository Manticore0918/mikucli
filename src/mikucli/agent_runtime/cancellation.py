from __future__ import annotations

from collections.abc import Callable


class StopRequested(RuntimeError):
    """Raised internally when the user requests that an active turn stop."""


def raise_if_stop_requested(stop_requested: Callable[[], bool] | None) -> None:
    if stop_requested is not None and stop_requested():
        raise StopRequested("stopped by user")

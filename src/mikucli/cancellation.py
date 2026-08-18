from __future__ import annotations

from collections.abc import Callable
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
from typing import Any


StopCallback = Callable[[], bool]
_current_stop_requested: ContextVar[StopCallback | None] = ContextVar(
    "mikucli_stop_requested",
    default=None,
)


class StopRequested(RuntimeError):
    """Raised internally when the user requests that an active turn stop."""


def raise_if_stop_requested(stop_requested: StopCallback | None) -> None:
    if stop_requested is not None and stop_requested():
        raise StopRequested("stopped by user")


def current_stop_requested() -> StopCallback | None:
    """Return the stop callback bound to the current agent turn."""

    return _current_stop_requested.get()


@contextmanager
def stop_context(stop_requested: StopCallback | None):
    token = _current_stop_requested.set(stop_requested)
    try:
        yield
    finally:
        _current_stop_requested.reset(token)


def binds_stop_context(function: Callable[..., Any]) -> Callable[..., Any]:
    """Bind a run_turn stop callback so nested model calls can be cancelled."""

    @wraps(function)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        with stop_context(kwargs.get("stop_requested")):
            return function(*args, **kwargs)

    return wrapped

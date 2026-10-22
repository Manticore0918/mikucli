from __future__ import annotations

from queue import Empty, Queue
from threading import Thread

from .run_control import check_active, current_budget


def approval_input(prompt: str) -> str:
    if current_budget() is None:
        return input(prompt)
    responses: Queue[str | BaseException] = Queue(maxsize=1)

    def read() -> None:
        try:
            responses.put(input(prompt))
        except (EOFError, OSError, KeyboardInterrupt) as exc:
            responses.put(exc)

    Thread(target=read, daemon=True).start()
    while True:
        check_active()
        try:
            answer = responses.get(timeout=0.05)
        except Empty:
            continue
        check_active()
        if isinstance(answer, BaseException):
            raise answer
        return answer

from __future__ import annotations

import io
import threading
from collections import deque
from collections.abc import Callable
from contextlib import redirect_stderr, redirect_stdout
from typing import Literal

from prompt_toolkit.application import Application
from prompt_toolkit.document import Document
from prompt_toolkit.input.base import Input
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import Layout
from prompt_toolkit.layout.containers import HSplit, Window, WindowRenderInfo
from prompt_toolkit.layout.controls import BufferControl, FormattedTextControl
from prompt_toolkit.layout.dimension import Dimension
from prompt_toolkit.output.base import Output
from prompt_toolkit.styles import Style
from prompt_toolkit.widgets import TextArea


TuiAction = Literal["continue", "fullscreen", "inline", "exit"]
SubmitHandler = Callable[[str], TuiAction]


class ActivityBuffer:
    """Keep a bounded number of rendered lines for the full-screen activity view."""

    def __init__(self, max_lines: int = 10_000) -> None:
        if max_lines < 1:
            raise ValueError("max_lines must be at least 1")
        self.max_lines = max_lines
        self._lines: deque[str] = deque()
        self._partial = ""
        self._lock = threading.RLock()

    def append(self, text: str) -> int:
        """Append text and return the number of leading characters evicted."""

        if not text:
            return 0
        with self._lock:
            pieces = (self._partial + text).splitlines(keepends=True)
            self._partial = ""
            if pieces and not pieces[-1].endswith(("\n", "\r")):
                self._partial = pieces.pop()
            dropped = 0
            for piece in pieces:
                self._lines.append(piece)
                while len(self._lines) > self.max_lines:
                    dropped += len(self._lines.popleft())
            if self._partial and len(self._lines) >= self.max_lines:
                dropped += len(self._lines.popleft())
            return dropped

    def text(self) -> str:
        with self._lock:
            return "".join(self._lines) + self._partial


class TuiOutputStream(io.TextIOBase):
    """A text stream that appends process output to the Terminal UI."""

    def __init__(self, write_handler: Callable[[str], None]) -> None:
        super().__init__()
        self._write_handler = write_handler

    @property
    def encoding(self) -> str:
        return "utf-8"

    def writable(self) -> bool:
        return True

    def isatty(self) -> bool:
        return True

    def write(self, text: str) -> int:
        if self.closed:
            raise ValueError("I/O operation on closed stream")
        if text:
            self._write_handler(text)
        return len(text)

    def flush(self) -> None:
        return None


class _ActivityWindow(Window):
    """A wrapped transcript window with reliable mouse-wheel scrolling."""

    @staticmethod
    def _minimum_scroll_for_cursor(info: WindowRenderInfo, cursor_line: int) -> int:
        """Return the earliest top line that keeps the cursor line visible."""

        available_height = info.window_height - info.configured_scroll_offsets.bottom
        used_height = 0
        previous_line = cursor_line
        for line_number in range(cursor_line, -1, -1):
            used_height += info.get_height_for_line(line_number)
            if used_height > available_height:
                return previous_line
            previous_line = line_number
        return 0

    @staticmethod
    def _maximum_scroll_for_cursor(info: WindowRenderInfo, cursor_line: int) -> int:
        """Return the latest top line that keeps the cursor line visible."""

        used_height = 0
        previous_line = cursor_line
        for line_number in range(cursor_line - 1, -1, -1):
            used_height += info.get_height_for_line(line_number)
            if used_height > info.configured_scroll_offsets.top:
                return previous_line
            previous_line = line_number
        return previous_line

    def _cursor_line(self, info: WindowRenderInfo) -> int:
        if isinstance(self.content, BufferControl):
            return self.content.buffer.document.cursor_position_row
        return info.ui_content.cursor_position.y

    def _scroll_up(self) -> None:
        info = self.render_info
        if info is None or self.vertical_scroll <= 0:
            return

        requested_scroll = self.vertical_scroll - 1
        cursor_line = self._cursor_line(info)

        # Window._scroll_when_linewrapping keeps the complete cursor line in
        # view. Move the hidden transcript cursor only when that constraint
        # would otherwise undo this wheel step during the next render.
        while cursor_line > 0 and requested_scroll < self._minimum_scroll_for_cursor(
            info, cursor_line
        ):
            self.content.move_cursor_up()
            cursor_line -= 1

        self.vertical_scroll = requested_scroll

    def _scroll_down(self) -> None:
        info = self.render_info
        if info is None:
            return

        # prompt-toolkit 3.0.52 compares logical lines with the window height
        # here. That prevents scrolling down when wrapping makes a small
        # number of logical lines taller than the viewport.
        remaining_height = -self.vertical_scroll_2
        for line_number in range(self.vertical_scroll, info.content_height):
            remaining_height += info.get_height_for_line(line_number)
            if remaining_height > info.window_height:
                break
        else:
            return

        requested_scroll = self.vertical_scroll + 1
        cursor_line = self._cursor_line(info)
        while (
            cursor_line < info.content_height - 1
            and requested_scroll > self._maximum_scroll_for_cursor(info, cursor_line)
        ):
            self.content.move_cursor_down()
            cursor_line += 1
        self.vertical_scroll = requested_scroll


class FullScreenTerminalUI:
    """A prompt-toolkit alternate-screen renderer with a fixed input composer."""

    def __init__(
        self,
        *,
        prompt_label: Callable[[], str],
        status_text: Callable[[], str],
        approval_text: Callable[[], str],
        is_busy: Callable[[], bool],
        max_activity_lines: int = 10_000,
        input: Input | None = None,
        output: Output | None = None,
    ) -> None:
        self._prompt_label = prompt_label
        self._status_text = status_text
        self._approval_text = approval_text
        self._is_busy = is_busy
        self._submit_handler: SubmitHandler | None = None
        self._activity_state = ActivityBuffer(max_activity_lines)
        self._activity_lock = threading.RLock()

        self.activity = TextArea(
            text="",
            read_only=True,
            focusable=True,
            focus_on_click=True,
            scrollbar=False,
            wrap_lines=True,
            style="class:activity",
        )
        self.activity.window = _ActivityWindow(
            content=self.activity.control,
            height=Dimension(min=1),
            wrap_lines=True,
            style="class:text-area class:activity",
        )
        self.composer = TextArea(
            text="",
            multiline=False,
            accept_handler=self._accept,
            prompt=lambda: [("class:prompt", self._prompt_label())],
            height=lambda: Dimension(min=1, max=8, preferred=min(8, self.composer.document.line_count)),
            focus_on_click=True,
            wrap_lines=True,
            style="class:composer",
        )
        status = Window(
            FormattedTextControl(lambda: [("class:status", f" {self._status_text()} ")]),
            height=1,
            style="class:status",
        )
        approval = Window(
            FormattedTextControl(lambda: [("class:approval", f" {self._approval_text()} ")]),
            height=lambda: 1 if self._approval_text() else 0,
            style="class:approval",
        )
        separator = Window(height=1, char="─", style="class:separator")
        root = HSplit([self.activity, separator, approval, status, self.composer])

        self.key_bindings = self._create_key_bindings()
        self.application: Application[TuiAction] = Application(
            layout=Layout(root, focused_element=self.composer),
            key_bindings=self.key_bindings,
            full_screen=True,
            mouse_support=True,
            enable_page_navigation_bindings=False,
            refresh_interval=0.1,
            input=input,
            output=output,
            style=Style.from_dict(
                {
                    "activity": "",
                    "separator": "fg:#666666",
                    "status": "reverse",
                    "approval": "fg:#ffcc00 bold",
                    "prompt": "fg:#66d9ef bold",
                    "composer": "",
                }
            ),
        )
        self.stream = TuiOutputStream(self.append_output)

    def append_output(self, text: str) -> None:
        with self._activity_lock:
            old_document = self.activity.buffer.document
            was_at_bottom = old_document.cursor_position >= len(old_document.text)
            dropped = self._activity_state.append(text)
            new_text = self._activity_state.text()
            if was_at_bottom:
                cursor = len(new_text)
            else:
                cursor = max(0, min(len(new_text), old_document.cursor_position - dropped))
            self.activity.buffer.set_document(Document(new_text, cursor), bypass_readonly=True)
        self.application.invalidate()

    def run(self, submit_handler: SubmitHandler) -> TuiAction:
        self._submit_handler = submit_handler
        with redirect_stdout(self.stream), redirect_stderr(self.stream):
            return self.application.run()

    def _accept(self, buffer) -> bool:
        prompt = buffer.text.strip()
        if not prompt:
            return True
        buffer.reset()
        self._dispatch(prompt)
        return True

    def _dispatch(self, prompt: str) -> None:
        handler = self._submit_handler
        if handler is None:
            return
        self.append_output(f"{self._prompt_label()}{prompt}\n")
        action = handler(prompt)
        if action != "continue" and self.application.is_running:
            self.application.exit(result=action)
        else:
            self.application.invalidate()

    def _create_key_bindings(self) -> KeyBindings:
        bindings = KeyBindings()

        @bindings.add("escape", "enter")
        def insert_newline(event) -> None:
            event.current_buffer.insert_text("\n")

        @bindings.add("c-c")
        def interrupt(event) -> None:
            if self._is_busy():
                self._dispatch("/stop")
                return
            event.app.layout.focus(self.composer)
            self.composer.buffer.reset()

        @bindings.add("c-d")
        def exit_session(event) -> None:
            if self._is_busy():
                return
            if self.composer.text:
                self.composer.buffer.delete()
                return
            self._dispatch("/exit")

        @bindings.add("pageup")
        def page_up(event) -> None:
            count = max(1, event.app.output.get_size().rows - 4)
            self.activity.buffer.cursor_up(count=count)
            event.app.invalidate()

        @bindings.add("pagedown")
        def page_down(event) -> None:
            count = max(1, event.app.output.get_size().rows - 4)
            self.activity.buffer.cursor_down(count=count)
            event.app.invalidate()

        return bindings

from __future__ import annotations

import unittest

from prompt_toolkit.input.defaults import create_pipe_input
from prompt_toolkit.output import DummyOutput

from mikucli.terminal_ui import ActivityBuffer, FullScreenTerminalUI, TuiOutputStream


class ActivityBufferTests(unittest.TestCase):
    def test_retains_only_the_configured_number_of_lines(self) -> None:
        activity = ActivityBuffer(max_lines=3)

        dropped = activity.append("one\ntwo\nthree\nfour\n")

        self.assertEqual(activity.text(), "two\nthree\nfour\n")
        self.assertEqual(dropped, len("one\n"))

    def test_partial_writes_form_one_rendered_line(self) -> None:
        activity = ActivityBuffer(max_lines=2)

        activity.append("hel")
        activity.append("lo\nworld")

        self.assertEqual(activity.text(), "hello\nworld")


class TuiOutputStreamTests(unittest.TestCase):
    def test_forwards_text_and_reports_the_written_length(self) -> None:
        writes: list[str] = []
        stream = TuiOutputStream(writes.append)

        written = stream.write("hello")

        self.assertEqual(written, 5)
        self.assertEqual(writes, ["hello"])


class FullScreenTerminalUITests(unittest.TestCase):
    def test_submitted_text_reaches_handler_and_exit_closes_application(self) -> None:
        prompts: list[str] = []

        def submit(prompt: str):
            prompts.append(prompt)
            return "exit" if prompt == "/exit" else "continue"

        with create_pipe_input() as pipe_input:
            ui = FullScreenTerminalUI(
                prompt_label=lambda: "You: ",
                status_text=lambda: "Agent: idle",
                approval_text=lambda: "",
                is_busy=lambda: False,
                input=pipe_input,
                output=DummyOutput(),
            )
            pipe_input.send_text("hello\r/exit\r")

            action = ui.run(submit)

        self.assertEqual(action, "exit")
        self.assertEqual(prompts, ["hello", "/exit"])
        self.assertIn("You: hello", ui.activity.text)

    def test_ctrl_c_dispatches_stop_while_busy(self) -> None:
        prompts: list[str] = []

        def submit(prompt: str):
            prompts.append(prompt)
            return "exit"

        with create_pipe_input() as pipe_input:
            ui = FullScreenTerminalUI(
                prompt_label=lambda: "You: ",
                status_text=lambda: "Agent: thinking",
                approval_text=lambda: "",
                is_busy=lambda: True,
                input=pipe_input,
                output=DummyOutput(),
            )
            pipe_input.send_bytes(b"\x03")

            action = ui.run(submit)

        self.assertEqual(action, "exit")
        self.assertEqual(prompts, ["/stop"])

    def test_alt_enter_inserts_a_newline_without_submitting(self) -> None:
        prompts: list[str] = []

        def submit(prompt: str):
            prompts.append(prompt)
            return "exit"

        with create_pipe_input() as pipe_input:
            ui = FullScreenTerminalUI(
                prompt_label=lambda: "You: ",
                status_text=lambda: "Agent: idle",
                approval_text=lambda: "",
                is_busy=lambda: False,
                input=pipe_input,
                output=DummyOutput(),
            )
            pipe_input.send_text("first")
            pipe_input.send_bytes(b"\x1b\r")
            pipe_input.send_text("second\r")

            action = ui.run(submit)

        self.assertEqual(action, "exit")
        self.assertEqual(prompts, ["first\nsecond"])


if __name__ == "__main__":
    unittest.main()

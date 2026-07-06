from __future__ import annotations

import io
import unittest
from contextlib import redirect_stdout

from mikucli.cli import build_parser, configure_output_encoding, handle_slash_command, render_banner
from mikucli.console import TerminalConsole


class CliTests(unittest.TestCase):
    def test_parser_accepts_task_workspace_and_model(self) -> None:
        args = build_parser().parse_args(["hello", "--workspace", ".", "--model", "glm-5.2"])
        self.assertEqual(args.task_prompt, ["hello"])
        self.assertEqual(args.workspace, ".")
        self.assertEqual(args.model, "glm-5.2")

    def test_parser_accepts_env_file_path(self) -> None:
        args = build_parser().parse_args(["--env-file", "mikucli.env"])
        self.assertEqual(args.env_file, "mikucli.env")

    def test_parser_accepts_context_window_tokens(self) -> None:
        args = build_parser().parse_args(["--context-window-tokens", "64000"])
        self.assertEqual(args.context_window_tokens, 64000)

    def test_banner_renders_boxed_art(self) -> None:
        lines = render_banner().splitlines()
        self.assertTrue(lines[0].startswith("╔"))
        self.assertTrue(lines[-1].startswith("╚"))
        self.assertIn("███╗", render_banner())
        self.assertIn("███████╗", render_banner())
        self.assertEqual(len({len(line) for line in lines}), 1)

    def test_configure_output_encoding_is_safe_to_call(self) -> None:
        configure_output_encoding()

    def test_language_slash_commands_switch_console_language(self) -> None:
        console = TerminalConsole()
        buffer = io.StringIO()

        with redirect_stdout(buffer):
            self.assertTrue(handle_slash_command("/lang-chn", _UnusedCodebaseService(), console))
            self.assertEqual(console.language, "chn")
            self.assertEqual(console.prompt_label(), "你: ")

            self.assertTrue(handle_slash_command("/lang-eng", _UnusedCodebaseService(), console))
            self.assertEqual(console.language, "eng")
            self.assertEqual(console.prompt_label(), "You: ")

        self.assertIn("界面语言已切换为中文", buffer.getvalue())
        self.assertIn("language switched to English", buffer.getvalue())


class _UnusedCodebaseService:
    pass


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import unittest

from mikucli.cli import build_parser, configure_output_encoding, render_banner


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
        self.assertTrue(lines[0].startswith("+"))
        self.assertTrue(lines[-1].startswith("+"))
        self.assertIn("mikucli", render_banner())
        self.assertEqual(len({len(line) for line in lines}), 1)

    def test_configure_output_encoding_is_safe_to_call(self) -> None:
        configure_output_encoding()


if __name__ == "__main__":
    unittest.main()

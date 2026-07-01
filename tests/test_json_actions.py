from __future__ import annotations

import unittest

from mikucli.json_actions import parse_json_action


class JsonActionTests(unittest.TestCase):
    def test_parses_tool_action(self) -> None:
        action = parse_json_action('{"tool":"read_file","arguments":{"path":"README.md"}}')
        self.assertIsNotNone(action)
        assert action is not None
        self.assertEqual(action.kind, "tool")
        self.assertEqual(action.name, "read_file")
        self.assertEqual(action.arguments["path"], "README.md")

    def test_parses_final_action(self) -> None:
        action = parse_json_action('{"final":"done"}')
        self.assertIsNotNone(action)
        assert action is not None
        self.assertEqual(action.kind, "final")
        self.assertEqual(action.arguments["content"], "done")


if __name__ == "__main__":
    unittest.main()

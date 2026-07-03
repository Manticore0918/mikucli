from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from mikucli.mcp_config import McpConfigError, load_mcp_config
from mikucli.tools import ToolRiskLevel


class McpConfigTests(unittest.TestCase):
    def test_loads_servers_and_tool_bindings(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_path = root / ".mikucli" / "mcp.json"
            config_path.parent.mkdir()
            config_path.write_text(
                """
                {
                  "servers": {
                    "zread": {
                      "command": "zread-mcp",
                      "args": ["--stdio"],
                      "env": {"ZREAD_TOKEN": "token"}
                    }
                  },
                  "tools": {
                    "read_github_file": {
                      "server": "zread",
                      "mcp_tool_name": "read_file",
                      "risk": "low"
                    }
                  }
                }
                """,
                encoding="utf-8",
            )

            config = load_mcp_config(root)

            self.assertEqual(config.servers["zread"].command, "zread-mcp")
            self.assertEqual(config.servers["zread"].args, ["--stdio"])
            self.assertEqual(config.servers["zread"].env, {"ZREAD_TOKEN": "token"})
            binding = config.tools["read_github_file"]
            self.assertEqual(binding.internal_id, "zread.read_file")
            self.assertEqual(binding.risk, ToolRiskLevel.LOW)

    def test_risk_defaults_to_high(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_path = root / ".mikucli" / "mcp.json"
            config_path.parent.mkdir()
            config_path.write_text(
                """
                {
                  "servers": {"filesystem": {"command": "npx"}},
                  "tools": {
                    "read_workspace_file": {
                      "server": "filesystem",
                      "mcp_tool_name": "read_file"
                    }
                  }
                }
                """,
                encoding="utf-8",
            )

            config = load_mcp_config(root)

            self.assertEqual(config.tools["read_workspace_file"].risk, ToolRiskLevel.HIGH)

    def test_rejects_unknown_server_binding(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_path = root / ".mikucli" / "mcp.json"
            config_path.parent.mkdir()
            config_path.write_text(
                """
                {
                  "servers": {"filesystem": {"command": "npx"}},
                  "tools": {
                    "read_github_file": {
                      "server": "zread",
                      "mcp_tool_name": "read_file"
                    }
                  }
                }
                """,
                encoding="utf-8",
            )

            with self.assertRaisesRegex(McpConfigError, "unknown server"):
                load_mcp_config(root)

    def test_missing_config_has_clear_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(McpConfigError, "MCP config not found"):
                load_mcp_config(Path(tmp))


if __name__ == "__main__":
    unittest.main()

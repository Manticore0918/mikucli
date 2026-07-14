from __future__ import annotations

import io
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

from mikucli.config import ConfigError
from mikucli.console import TerminalConsole
from mikucli.llm import TokenUsage
from mikucli.tools import ToolApprovalRequest, ToolRiskLevel


class ConsoleTests(unittest.TestCase):
    def test_progress_uses_thinking_label(self) -> None:
        output = _capture(lambda console: console.progress("Thinking...."))
        self.assertIn("🤔Thinking....", output)

    def test_answer_uses_agent_label(self) -> None:
        output = _capture(lambda console: console.answer("hello"))
        self.assertIn("🤖Agent: hello", output)

    def test_tool_request_uses_tools_label(self) -> None:
        output = _capture(lambda console: console.tool_request("read_file", {"path": "README.md"}))
        self.assertIn("🔧Tools: read_file", output)

    def test_token_usage_uses_token_label(self) -> None:
        output = _capture(
            lambda console: console.token_usage(
                TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15)
            )
        )
        self.assertIn("📊Token: total=15, prompt=10, completion=5", output)

    def test_sensitive_file_approval_uses_read_specific_prompt(self) -> None:
        request = ToolApprovalRequest(
            tool_name="read_file",
            risk_level=ToolRiskLevel.MEDIUM,
            workspace="workspace",
            summary="Read sensitive file: .env",
        )
        console = TerminalConsole()

        with patch("builtins.input", return_value="yes") as prompt:
            output = _capture_existing(console, lambda: self.assertTrue(console.confirm_tool(request)))

        prompt.assert_called_once_with("Read this sensitive file? [y/N] ")
        self.assertIn("Read sensitive file: .env", output)

    def test_chinese_language_localizes_console_chrome(self) -> None:
        console = TerminalConsole(language="chn")

        self.assertEqual(console.prompt_label(), "你: ")
        output = _capture_existing(console, lambda: console.progress("Thinking...."))
        self.assertIn("🤔思考中....", output)
        output = _capture_existing(console, lambda: console.answer("hello"))
        self.assertIn("🤖智能体: hello", output)
        output = _capture_existing(console, lambda: console.tool_result("read_file", True, "ok"))
        self.assertIn("🔧工具: read_file -> 成功", output)
        output = _capture_existing(console, lambda: console.log_path("run.json"))
        self.assertIn("[日志] run.json", output)
        self.assertEqual(console.error(ValueError("bad")), "mikucli：bad")


    def test_config_error_uses_localized_console_message(self) -> None:
        error = ConfigError("created user config template", "已创建用户配置模板")

        self.assertEqual(TerminalConsole(language="eng").error(error), "mikucli: created user config template")
        self.assertEqual(TerminalConsole(language="chn").error(error), "mikucli：已创建用户配置模板")


def _capture(action) -> str:
    console = TerminalConsole()
    return _capture_existing(console, lambda: action(console))


def _capture_existing(console: TerminalConsole, action) -> str:
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        action()
    return buffer.getvalue()


if __name__ == "__main__":
    unittest.main()

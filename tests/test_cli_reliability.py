from __future__ import annotations

import json
import os
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from mikucli.cli import build_parser, main
from mikucli.config import load_config
from mikucli.llm import AssistantMessage, TokenUsage, ToolCall
from mikucli.runtime_config import runtime_settings


class CliReliabilityTests(unittest.TestCase):
    def run_cli(self, responses, *flags):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        output, errors = StringIO(), StringIO()
        with patch("mikucli.cli.ModelClient") as constructor:
            constructor.return_value.chat.side_effect = responses
            with redirect_stdout(output), redirect_stderr(errors):
                code = main(
                    [
                        "--workspace",
                        temporary.name,
                        "--api-key",
                        "dummy",
                        "--json",
                        *flags,
                        "do work",
                    ]
                )
        return code, json.loads(output.getvalue()), errors.getvalue()

    def test_json_stdout_contains_only_one_result_and_real_exit_code(self):
        response = AssistantMessage("done", [], {}, TokenUsage(total_tokens=3))
        code, result, diagnostic = self.run_cli([response])
        self.assertEqual(code, 0)
        self.assertEqual(result["status"], "succeeded")
        self.assertEqual(result["schema_version"], 1)
        self.assertTrue(Path(result["log_path"]).is_file())
        self.assertIn("Model:", diagnostic)
        tool = AssistantMessage(
            "", [ToolCall("call", "list_files", {})], {}, TokenUsage()
        )
        code, result, _ = self.run_cli([tool], "--max-steps", "1")
        self.assertEqual(code, 3)
        self.assertEqual(result["reason"], "max_steps")

    def test_json_approval_is_denied_without_reading_stdin(self):
        call = ToolCall("call", "write_file", {"path": "denied.txt", "content": "x"})
        responses = [
            AssistantMessage("", [call], {}, TokenUsage()),
            AssistantMessage("blocked", [], {}, TokenUsage()),
        ]
        with patch("builtins.input", side_effect=AssertionError("must not read stdin")):
            code, result, _ = self.run_cli(responses)
        self.assertEqual(code, 1)
        self.assertEqual(result["reason"], "approval_denied")

    def test_runtime_configuration_errors_return_json_and_exit_two(self):
        code, result, _ = self.run_cli([], "--max-duration", "nan")
        self.assertEqual(code, 2)
        self.assertEqual(result["reason"], "invalid_configuration")
        code, result, _ = self.run_cli([], "--max-spend", "1")
        self.assertEqual(code, 2)
        self.assertIn("requires both", result["error"]["message"])

    def test_positive_steps_are_required(self):
        for value in ("0", "-1"):
            with (
                self.subTest(value=value),
                redirect_stderr(StringIO()),
                self.assertRaises(SystemExit) as stopped,
            ):
                build_parser().parse_args(["--max-steps", value])
            self.assertEqual(stopped.exception.code, 2)

    def test_settings_merge_file_environment_and_cli_by_key(self):
        with patch.dict(os.environ, {"MIKUCLI_MAX_RETRIES": "1"}):
            policy, limits = runtime_settings(
                {"MIKUCLI_MAX_RETRIES": "3", "MIKUCLI_MAX_TOKENS": "99"},
                {"request_timeout": 7},
            )
            self.assertEqual(policy.max_retries, 1)
            self.assertEqual(policy.timeout_seconds, 7)
            self.assertEqual(limits.max_tokens, 99)
            policy, _ = runtime_settings({}, {"max_retries": 0})
            self.assertEqual(policy.max_retries, 0)

    def test_model_endpoint_is_validated_before_dispatch(self):
        with (
            tempfile.TemporaryDirectory() as directory,
            self.assertRaisesRegex(ValueError, "HTTP"),
        ):
            load_config(
                Path(directory), "test", api_key="dummy", base_url="file:///bad"
            )

    def test_dependency_check_reports_failed_ollama_without_chat_call(self):
        with tempfile.TemporaryDirectory() as directory:
            output, errors = StringIO(), StringIO()
            with (
                patch("mikucli.cli.ModelClient") as client,
                patch("mikucli.cli_support.reliability.request_json") as request,
            ):
                request.side_effect = RuntimeError("offline")
                with redirect_stdout(output), redirect_stderr(errors):
                    code = main(
                        [
                            "--workspace",
                            directory,
                            "--api-key",
                            "dummy",
                            "--check",
                            "--json",
                        ]
                    )
            self.assertEqual(code, 1)
            report = json.loads(output.getvalue())
            self.assertEqual(report["reason"], "dependency_unavailable")
            self.assertFalse(
                next(
                    item
                    for item in report["checks"]
                    if item["name"] == "embedding_service"
                )["ok"]
            )
            client.return_value.chat.assert_not_called()


if __name__ == "__main__":
    unittest.main()

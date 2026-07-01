from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from mikucli.config import (
    DEFAULT_BASE_URL,
    DEFAULT_CONTEXT_WINDOW_TOKENS,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_EMBEDDING_PROVIDER,
    DEFAULT_MODEL,
    DEFAULT_OLLAMA_BASE_URL,
    load_config,
)


class ConfigTests(unittest.TestCase):
    def test_loads_api_key_from_workspace_env_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / ".env").write_text(
                'BIGMODEL_API_KEY="from-file"\n',
                encoding="utf-8",
            )

            with patch.dict(os.environ, {}, clear=True):
                config = load_config(root, model=None)

            self.assertEqual(config.api_key, "from-file")
            self.assertEqual(config.model, DEFAULT_MODEL)
            self.assertEqual(config.base_url, DEFAULT_BASE_URL)
            self.assertEqual(config.context_window_tokens, DEFAULT_CONTEXT_WINDOW_TOKENS)
            self.assertEqual(config.embedding_provider, DEFAULT_EMBEDDING_PROVIDER)
            self.assertEqual(config.embedding_model, DEFAULT_EMBEDDING_MODEL)
            self.assertEqual(config.ollama_base_url, DEFAULT_OLLAMA_BASE_URL)

    def test_environment_overrides_env_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / ".env").write_text(
                'BIGMODEL_API_KEY=from-file\nMIKUCLI_MODEL=file-model\n',
                encoding="utf-8",
            )

            with patch.dict(
                os.environ,
                {"BIGMODEL_API_KEY": "from-env", "MIKUCLI_MODEL": "env-model"},
                clear=True,
            ):
                config = load_config(root, model=None)

            self.assertEqual(config.api_key, "from-env")
            self.assertEqual(config.model, "env-model")

    def test_cli_model_overrides_environment_and_env_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / ".env").write_text(
                'BIGMODEL_API_KEY=from-file\nMIKUCLI_MODEL=file-model\n',
                encoding="utf-8",
            )

            with patch.dict(os.environ, {"MIKUCLI_MODEL": "env-model"}, clear=True):
                config = load_config(root, model="cli-model")

            self.assertEqual(config.model, "cli-model")

    def test_accepts_explicit_env_file_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            custom_env = root / "custom.env"
            custom_env.write_text("BIGMODEL_API_KEY=custom-key\n", encoding="utf-8")

            with patch.dict(os.environ, {}, clear=True):
                config = load_config(root, model=None, env_file=custom_env)

            self.assertEqual(config.api_key, "custom-key")
            self.assertEqual(config.env_file, custom_env.resolve())

    def test_env_file_from_environment(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            custom_env = root / "custom.env"
            custom_env.write_text("BIGMODEL_API_KEY=custom-key\n", encoding="utf-8")

            with patch.dict(os.environ, {"MIKUCLI_ENV_FILE": str(custom_env)}, clear=True):
                config = load_config(root, model=None)

            self.assertEqual(config.api_key, "custom-key")

    def test_context_window_tokens_can_come_from_env_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / ".env").write_text(
                "BIGMODEL_API_KEY=from-file\nMIKUCLI_CONTEXT_WINDOW_TOKENS=64000\n",
                encoding="utf-8",
            )

            with patch.dict(os.environ, {}, clear=True):
                config = load_config(root, model=None)

            self.assertEqual(config.context_window_tokens, 64000)

    def test_cli_context_window_tokens_overrides_environment_and_env_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / ".env").write_text(
                "BIGMODEL_API_KEY=from-file\nMIKUCLI_CONTEXT_WINDOW_TOKENS=64000\n",
                encoding="utf-8",
            )

            with patch.dict(os.environ, {"MIKUCLI_CONTEXT_WINDOW_TOKENS": "32000"}, clear=True):
                config = load_config(root, model=None, context_window_tokens=16000)

            self.assertEqual(config.context_window_tokens, 16000)

    def test_embedding_config_can_come_from_env_file_and_environment(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / ".env").write_text(
                "\n".join(
                    [
                        "BIGMODEL_API_KEY=from-file",
                        "MIKUCLI_EMBEDDING_PROVIDER=ollama",
                        "MIKUCLI_EMBEDDING_MODEL=file-embed",
                        "MIKUCLI_OLLAMA_BASE_URL=http://file-host:11434",
                    ]
                ),
                encoding="utf-8",
            )

            with patch.dict(
                os.environ,
                {
                    "MIKUCLI_EMBEDDING_MODEL": "env-embed",
                    "MIKUCLI_OLLAMA_BASE_URL": "http://env-host:11434/",
                },
                clear=True,
            ):
                config = load_config(root, model=None)

            self.assertEqual(config.embedding_provider, "ollama")
            self.assertEqual(config.embedding_model, "env-embed")
            self.assertEqual(config.ollama_base_url, "http://env-host:11434")


if __name__ == "__main__":
    unittest.main()

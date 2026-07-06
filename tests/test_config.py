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
    DEFAULT_ENV_TEMPLATE,
    DEFAULT_MODEL,
    DEFAULT_OLLAMA_BASE_URL,
    default_user_env_file,
    load_config,
)


class ConfigTests(unittest.TestCase):
    def setUp(self) -> None:
        self._home_tmp = tempfile.TemporaryDirectory()
        self._home_patcher = patch.object(Path, "home", return_value=Path(self._home_tmp.name))
        self._home_patcher.start()

    def tearDown(self) -> None:
        self._home_patcher.stop()
        self._home_tmp.cleanup()

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

    def test_loads_api_key_from_user_config_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            home = root / "home"
            user_env = home / ".mikucli" / ".env"
            user_env.parent.mkdir(parents=True)
            user_env.write_text("BIGMODEL_API_KEY=from-user\n", encoding="utf-8")

            with patch.object(Path, "home", return_value=home):
                with patch.dict(os.environ, {}, clear=True):
                    config = load_config(root, model=None)

            self.assertEqual(config.api_key, "from-user")
            self.assertEqual(config.env_file, user_env.resolve())

    def test_workspace_env_overrides_user_config_by_key(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            home = root / "home"
            user_env = home / ".mikucli" / ".env"
            user_env.parent.mkdir(parents=True)
            user_env.write_text(
                "BIGMODEL_API_KEY=from-user\nMIKUCLI_MODEL=user-model\n",
                encoding="utf-8",
            )
            (root / ".env").write_text("MIKUCLI_MODEL=workspace-model\n", encoding="utf-8")

            with patch.object(Path, "home", return_value=home):
                with patch.dict(os.environ, {}, clear=True):
                    config = load_config(root, model=None)

            self.assertEqual(config.api_key, "from-user")
            self.assertEqual(config.model, "workspace-model")
            self.assertEqual(config.env_file, (root / ".env").resolve())

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

    def test_explicit_env_files_merge_by_key_in_priority_order(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            home = root / "home"
            user_env = home / ".mikucli" / ".env"
            user_env.parent.mkdir(parents=True)
            user_env.write_text(
                "BIGMODEL_API_KEY=from-user\nMIKUCLI_MODEL=user-model\nBIGMODEL_BASE_URL=http://user\n",
                encoding="utf-8",
            )
            (root / ".env").write_text(
                "MIKUCLI_MODEL=workspace-model\nBIGMODEL_BASE_URL=http://workspace\n",
                encoding="utf-8",
            )
            env_file_from_env = root / "env-file.env"
            env_file_from_env.write_text("BIGMODEL_BASE_URL=http://env-file\n", encoding="utf-8")
            cli_env_file = root / "cli.env"
            cli_env_file.write_text("MIKUCLI_CONTEXT_WINDOW_TOKENS=64000\n", encoding="utf-8")

            with patch.object(Path, "home", return_value=home):
                with patch.dict(os.environ, {"MIKUCLI_ENV_FILE": str(env_file_from_env)}, clear=True):
                    config = load_config(root, model=None, env_file=cli_env_file)

            self.assertEqual(config.api_key, "from-user")
            self.assertEqual(config.model, "workspace-model")
            self.assertEqual(config.base_url, "http://env-file")
            self.assertEqual(config.context_window_tokens, 64000)
            self.assertEqual(config.env_file, cli_env_file.resolve())

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

    def test_missing_api_key_creates_user_config_template(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            home = root / "home"

            with patch.object(Path, "home", return_value=home):
                with patch.dict(os.environ, {}, clear=True):
                    with self.assertRaises(ValueError) as raised:
                        load_config(root, model=None)
                user_env = default_user_env_file()

            self.assertIn("created user config template", str(raised.exception))
            self.assertEqual(user_env.read_text(encoding="utf-8"), DEFAULT_ENV_TEMPLATE)

    def test_missing_explicit_env_file_does_not_create_user_config_template(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            home = root / "home"

            with patch.object(Path, "home", return_value=home):
                with patch.dict(os.environ, {}, clear=True):
                    with self.assertRaises(ValueError) as raised:
                        load_config(root, model=None, env_file=root / "missing.env")
                user_env = default_user_env_file()

            self.assertIn("env file path does not exist", str(raised.exception))
            self.assertFalse(user_env.exists())


if __name__ == "__main__":
    unittest.main()

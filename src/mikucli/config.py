from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


DEFAULT_MODEL = "glm-5.2"
DEFAULT_BASE_URL = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
DEFAULT_ENV_FILE = ".env"
USER_CONFIG_DIR = ".mikucli"
DEFAULT_CONTEXT_WINDOW_TOKENS = 128000
DEFAULT_EMBEDDING_PROVIDER = "ollama"
DEFAULT_EMBEDDING_MODEL = "nomic-embed-text"
DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_ENV_TEMPLATE = """# Required: replace with your BigModel API key.
BIGMODEL_API_KEY=

# Optional defaults.
MIKUCLI_MODEL=glm-5.2
BIGMODEL_BASE_URL=https://open.bigmodel.cn/api/paas/v4/chat/completions
MIKUCLI_CONTEXT_WINDOW_TOKENS=128000
MIKUCLI_EMBEDDING_PROVIDER=ollama
MIKUCLI_EMBEDDING_MODEL=nomic-embed-text
MIKUCLI_OLLAMA_BASE_URL=http://localhost:11434
"""


class ConfigError(ValueError):
    def __init__(self, english: str, chinese: str) -> None:
        super().__init__(english)
        self.english = english
        self.chinese = chinese

    def localized(self, language: str) -> str:
        return self.chinese if language == "chn" else self.english


@dataclass(frozen=True)
class Config:
    api_key: str
    model: str
    workspace: Path
    base_url: str = DEFAULT_BASE_URL
    env_file: Path | None = None
    context_window_tokens: int = DEFAULT_CONTEXT_WINDOW_TOKENS
    embedding_provider: str = DEFAULT_EMBEDDING_PROVIDER
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    ollama_base_url: str = DEFAULT_OLLAMA_BASE_URL


def load_config(
    workspace: Path,
    model: str | None,
    env_file: Path | None = None,
    context_window_tokens: int | None = None,
) -> Config:
    resolved_workspace = workspace.resolve()
    env_file_from_env = _env_file_from_env()
    user_env_file = default_user_env_file()
    file_env, loaded_env_files = _load_file_env(
        workspace=resolved_workspace,
        user_env_file=user_env_file,
        env_file_from_env=env_file_from_env,
        cli_env_file=env_file,
    )

    api_key = _first_present(
        os.environ.get("BIGMODEL_API_KEY"),
        file_env.get("BIGMODEL_API_KEY"),
    )
    selected_model = _first_present(
        model,
        os.environ.get("MIKUCLI_MODEL"),
        file_env.get("MIKUCLI_MODEL"),
        DEFAULT_MODEL,
    )
    base_url = _first_present(
        os.environ.get("BIGMODEL_BASE_URL"),
        file_env.get("BIGMODEL_BASE_URL"),
        DEFAULT_BASE_URL,
    )
    selected_context_window = _positive_int(
        _first_present(
            str(context_window_tokens) if context_window_tokens is not None else None,
            os.environ.get("MIKUCLI_CONTEXT_WINDOW_TOKENS"),
            file_env.get("MIKUCLI_CONTEXT_WINDOW_TOKENS"),
            str(DEFAULT_CONTEXT_WINDOW_TOKENS),
        ),
        "MIKUCLI_CONTEXT_WINDOW_TOKENS",
    )
    embedding_provider = _first_present(
        os.environ.get("MIKUCLI_EMBEDDING_PROVIDER"),
        file_env.get("MIKUCLI_EMBEDDING_PROVIDER"),
        DEFAULT_EMBEDDING_PROVIDER,
    )
    embedding_model = _first_present(
        os.environ.get("MIKUCLI_EMBEDDING_MODEL"),
        file_env.get("MIKUCLI_EMBEDDING_MODEL"),
        DEFAULT_EMBEDDING_MODEL,
    )
    ollama_base_url = _first_present(
        os.environ.get("MIKUCLI_OLLAMA_BASE_URL"),
        file_env.get("MIKUCLI_OLLAMA_BASE_URL"),
        DEFAULT_OLLAMA_BASE_URL,
    )

    if not api_key:
        if env_file is None and env_file_from_env is None and not user_env_file.exists():
            _create_default_user_env_file(user_env_file)
            raise ConfigError(
                f"created user config template at {user_env_file}. "
                "Fill BIGMODEL_API_KEY in that file, then restart mikucli.",
                f"已在 {user_env_file} 创建用户配置模板。请在该文件中填写 BIGMODEL_API_KEY，然后重启 mikucli。",
            )
        raise ConfigError(
            "BigModel API key is required. Set BIGMODEL_API_KEY in the environment "
            f"or in {user_env_file}, the workspace .env file, or an explicit --env-file.",
            "缺少 BigModel API key。请在环境变量 BIGMODEL_API_KEY、"
            f"{user_env_file}、工作区 .env 文件或显式 --env-file 中设置它。",
        )

    return Config(
        api_key=api_key,
        model=selected_model,
        workspace=resolved_workspace,
        base_url=base_url,
        env_file=loaded_env_files[-1] if loaded_env_files else None,
        context_window_tokens=selected_context_window,
        embedding_provider=embedding_provider,
        embedding_model=embedding_model,
        ollama_base_url=ollama_base_url.rstrip("/"),
    )


def default_user_env_file() -> Path:
    return (Path.home() / USER_CONFIG_DIR / DEFAULT_ENV_FILE).resolve()


def _load_file_env(
    *,
    workspace: Path,
    user_env_file: Path,
    env_file_from_env: Path | None,
    cli_env_file: Path | None,
) -> tuple[dict[str, str], list[Path]]:
    files = [
        (user_env_file, False),
        (_resolve_env_file(workspace, Path(DEFAULT_ENV_FILE)), False),
    ]
    if env_file_from_env is not None:
        files.append((_resolve_env_file(workspace, env_file_from_env), True))
    if cli_env_file is not None:
        files.append((_resolve_env_file(workspace, cli_env_file), True))

    values: dict[str, str] = {}
    loaded_files: list[Path] = []
    for path, required in files:
        file_values = _read_env_file(path, required=required)
        if path.exists():
            loaded_files.append(path)
        values.update(file_values)
    return values, loaded_files


def _resolve_env_file(workspace: Path, env_file: Path) -> Path:
    raw_path = env_file
    path = raw_path.expanduser()
    if path.is_absolute():
        return path.resolve()
    return (workspace / path).resolve()


def _env_file_from_env() -> Path | None:
    raw = os.environ.get("MIKUCLI_ENV_FILE", "").strip()
    return Path(raw) if raw else None


def _read_env_file(path: Path, *, required: bool = False) -> dict[str, str]:
    if not path.exists():
        if required:
            raise ConfigError(
                f"env file path does not exist: {path}",
                f"env 文件路径不存在：{path}",
            )
        return {}
    if not path.is_file():
        raise ConfigError(
            f"env file path is not a file: {path}",
            f"env 文件路径不是文件：{path}",
        )

    values: dict[str, str] = {}
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line.removeprefix("export ").strip()
        if "=" not in line:
            raise ConfigError(
                f"invalid .env line {line_number}: expected KEY=VALUE",
                f".env 第 {line_number} 行无效：应为 KEY=VALUE",
            )
        key, value = line.split("=", 1)
        key = key.strip()
        if not key:
            raise ConfigError(
                f"invalid .env line {line_number}: key is empty",
                f".env 第 {line_number} 行无效：key 为空",
            )
        values[key] = _clean_env_value(value)
    return values


def _create_default_user_env_file(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(DEFAULT_ENV_TEMPLATE, encoding="utf-8")


def _clean_env_value(value: str) -> str:
    cleaned = value.strip()
    if len(cleaned) >= 2 and cleaned[0] == cleaned[-1] and cleaned[0] in {'"', "'"}:
        return cleaned[1:-1]
    return cleaned


def _first_present(*values: str | None) -> str:
    for value in values:
        if value is not None and value.strip():
            return value.strip()
    return ""


def _positive_int(value: str, name: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    if parsed <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return parsed

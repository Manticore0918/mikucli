from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from .http_requests import RequestPolicy
from .run_control import RunLimits
from .runtime_config import runtime_settings, validate_endpoint


DEFAULT_MODEL = "glm-5.2"
DEFAULT_BASE_URL = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
DEFAULT_ENV_FILE = ".env"
USER_CONFIG_DIR = ".mikucli"
DEFAULT_CONTEXT_WINDOW_TOKENS = 1000000
DEFAULT_CONTEXT_SOFT_CAP_TOKENS = 450000
DEFAULT_EMBEDDING_PROVIDER = "ollama"
DEFAULT_EMBEDDING_MODEL = "nomic-embed-text"
DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_ENV_TEMPLATE = """# Required: API key for your model provider.
MIKUCLI_API_KEY=

# Model and OpenAI-compatible Chat Completions endpoint.
MIKUCLI_MODEL=glm-5.2
MIKUCLI_BASE_URL=https://open.bigmodel.cn/api/paas/v4/chat/completions

# Optional runtime defaults.
MIKUCLI_CONTEXT_WINDOW_TOKENS=1000000
MIKUCLI_CONTEXT_SOFT_CAP_TOKENS=450000
MIKUCLI_EMBEDDING_PROVIDER=ollama
MIKUCLI_EMBEDDING_MODEL=nomic-embed-text
MIKUCLI_OLLAMA_BASE_URL=http://localhost:11434
MIKUCLI_REQUEST_TIMEOUT_SECONDS=120
MIKUCLI_MAX_RETRIES=2
MIKUCLI_MAX_DURATION_SECONDS=900
MIKUCLI_MAX_OUTPUT_BYTES=1048576
"""


def resolve_context_soft_cap(
    context_window_tokens: int,
    context_soft_cap_tokens: int | None = None,
) -> int:
    if context_window_tokens <= 0:
        raise ValueError("context_window_tokens must be positive")
    if context_soft_cap_tokens is None:
        return min(DEFAULT_CONTEXT_SOFT_CAP_TOKENS, context_window_tokens)
    if context_soft_cap_tokens <= 0:
        raise ValueError("context_soft_cap_tokens must be positive")
    if context_soft_cap_tokens > context_window_tokens:
        raise ValueError(
            "context_soft_cap_tokens must not exceed context_window_tokens"
        )
    return context_soft_cap_tokens


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
    context_soft_cap_tokens: int = DEFAULT_CONTEXT_SOFT_CAP_TOKENS
    embedding_provider: str = DEFAULT_EMBEDDING_PROVIDER
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    ollama_base_url: str = DEFAULT_OLLAMA_BASE_URL
    request_policy: RequestPolicy = field(default_factory=RequestPolicy)
    run_limits: RunLimits = field(default_factory=RunLimits)


def load_config(
    workspace: Path,
    model: str | None,
    env_file: Path | None = None,
    context_window_tokens: int | None = None,
    *,
    context_soft_cap_tokens: int | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    runtime_overrides: dict[str, Any] | None = None,
) -> Config:
    resolved_workspace = workspace.resolve()
    env_file_from_env = _env_file_from_env()
    user_env_file = default_user_env_file()
    file_env, loaded_env_files, file_ranks = _load_file_env(
        workspace=resolved_workspace,
        user_env_file=user_env_file,
        env_file_from_env=env_file_from_env,
        cli_env_file=env_file,
    )

    selected_api_key = _first_present(
        api_key,
        os.environ.get("MIKUCLI_API_KEY"),
        os.environ.get("BIGMODEL_API_KEY"),
        file_env.get("MIKUCLI_API_KEY"),
    )
    selected_model = _first_present(
        model,
        os.environ.get("MIKUCLI_MODEL"),
        file_env.get("MIKUCLI_MODEL"),
        DEFAULT_MODEL,
    )
    selected_base_url = _first_present(
        base_url,
        os.environ.get("MIKUCLI_BASE_URL"),
        os.environ.get("BIGMODEL_BASE_URL"),
        file_env.get("MIKUCLI_BASE_URL"),
        DEFAULT_BASE_URL,
    )
    selected_context_window, hard_cap_rank = _select_positive_int_with_rank(
        cli_value=context_window_tokens,
        env_name="MIKUCLI_CONTEXT_WINDOW_TOKENS",
        file_env=file_env,
        file_ranks=file_ranks,
        default=DEFAULT_CONTEXT_WINDOW_TOKENS,
    )
    selected_context_soft_cap, soft_cap_rank = _select_positive_int_with_rank(
        cli_value=context_soft_cap_tokens,
        env_name="MIKUCLI_CONTEXT_SOFT_CAP_TOKENS",
        file_env=file_env,
        file_ranks=file_ranks,
        default=DEFAULT_CONTEXT_SOFT_CAP_TOKENS,
    )
    if selected_context_soft_cap > selected_context_window:
        if soft_cap_rank < hard_cap_rank:
            selected_context_soft_cap = selected_context_window
        else:
            raise ValueError(
                "MIKUCLI_CONTEXT_SOFT_CAP_TOKENS must not exceed MIKUCLI_CONTEXT_WINDOW_TOKENS"
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

    if not selected_api_key:
        if env_file is None and env_file_from_env is None and not user_env_file.exists():
            _create_default_user_env_file(user_env_file)
            raise ConfigError(
                f"created user config template at {user_env_file}. "
                "Fill MIKUCLI_API_KEY in that file, then restart mikucli.",
                f"已在 {user_env_file} 创建用户配置模板。请在该文件中填写 MIKUCLI_API_KEY，然后重启 mikucli。",
            )
        raise ConfigError(
            "A model provider API key is required. Set MIKUCLI_API_KEY in the environment "
            f"or in {user_env_file}, the workspace .env file, or an explicit --env-file.",
            "缺少模型提供商 API key。请在环境变量 MIKUCLI_API_KEY、"
            f"{user_env_file}、工作区 .env 文件或显式 --env-file 中设置它。",
        )

    validate_endpoint(selected_base_url, "MIKUCLI_BASE_URL")
    validate_endpoint(ollama_base_url, "MIKUCLI_OLLAMA_BASE_URL")
    policy, limits = runtime_settings(file_env, runtime_overrides)
    return Config(
        api_key=selected_api_key,
        model=selected_model,
        workspace=resolved_workspace,
        base_url=selected_base_url,
        env_file=loaded_env_files[-1] if loaded_env_files else None,
        context_window_tokens=selected_context_window,
        context_soft_cap_tokens=selected_context_soft_cap,
        embedding_provider=embedding_provider,
        embedding_model=embedding_model,
        ollama_base_url=ollama_base_url.rstrip("/"),
        request_policy=policy,
        run_limits=limits,
    )


def default_user_env_file() -> Path:
    return (Path.home() / USER_CONFIG_DIR / DEFAULT_ENV_FILE).resolve()


def _load_file_env(
    *,
    workspace: Path,
    user_env_file: Path,
    env_file_from_env: Path | None,
    cli_env_file: Path | None,
) -> tuple[dict[str, str], list[Path], dict[str, int]]:
    files = [
        (user_env_file, False),
        (_resolve_env_file(workspace, Path(DEFAULT_ENV_FILE)), False),
    ]
    if env_file_from_env is not None:
        files.append((_resolve_env_file(workspace, env_file_from_env), True))
    if cli_env_file is not None:
        files.append((_resolve_env_file(workspace, cli_env_file), True))

    values: dict[str, str] = {}
    ranks: dict[str, int] = {}
    loaded_files: list[Path] = []
    for rank, (path, required) in enumerate(files, start=1):
        file_values = _normalize_legacy_model_settings(_read_env_file(path, required=required))
        if path.exists():
            loaded_files.append(path)
        values.update(file_values)
        for key, value in file_values.items():
            if value.strip():
                ranks[key] = rank
    return values, loaded_files, ranks


def _normalize_legacy_model_settings(values: dict[str, str]) -> dict[str, str]:
    normalized = dict(values)
    aliases = (
        ("BIGMODEL_API_KEY", "MIKUCLI_API_KEY"),
        ("BIGMODEL_BASE_URL", "MIKUCLI_BASE_URL"),
    )
    for legacy_name, current_name in aliases:
        if not normalized.get(current_name, "").strip() and normalized.get(legacy_name, "").strip():
            normalized[current_name] = normalized[legacy_name]
    return normalized


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


def _select_positive_int_with_rank(
    *,
    cli_value: int | None,
    env_name: str,
    file_env: dict[str, str],
    file_ranks: dict[str, int],
    default: int,
) -> tuple[int, int]:
    if cli_value is not None:
        return _positive_int(str(cli_value), env_name), 6
    env_value = os.environ.get(env_name)
    if env_value is not None and env_value.strip():
        return _positive_int(env_value.strip(), env_name), 5
    file_value = file_env.get(env_name)
    if file_value is not None and file_value.strip():
        return _positive_int(file_value.strip(), env_name), file_ranks.get(env_name, 1)
    return default, 0

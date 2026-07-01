from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


DEFAULT_MODEL = "glm-5.2"
DEFAULT_BASE_URL = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
DEFAULT_ENV_FILE = ".env"
DEFAULT_CONTEXT_WINDOW_TOKENS = 128000
DEFAULT_EMBEDDING_PROVIDER = "ollama"
DEFAULT_EMBEDDING_MODEL = "nomic-embed-text"
DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"


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
    resolved_env_file = _resolve_env_file(resolved_workspace, env_file)
    file_env = _read_env_file(resolved_env_file)

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
        raise ValueError(
            "BigModel API key is required. Set BIGMODEL_API_KEY in the environment "
            "or in the workspace .env file."
        )

    return Config(
        api_key=api_key,
        model=selected_model,
        workspace=resolved_workspace,
        base_url=base_url,
        env_file=resolved_env_file if resolved_env_file.exists() else None,
        context_window_tokens=selected_context_window,
        embedding_provider=embedding_provider,
        embedding_model=embedding_model,
        ollama_base_url=ollama_base_url.rstrip("/"),
    )


def _resolve_env_file(workspace: Path, env_file: Path | None) -> Path:
    raw_path = env_file or _env_file_from_env() or Path(DEFAULT_ENV_FILE)
    path = raw_path.expanduser()
    if path.is_absolute():
        return path.resolve()
    return (workspace / path).resolve()


def _env_file_from_env() -> Path | None:
    raw = os.environ.get("MIKUCLI_ENV_FILE", "").strip()
    return Path(raw) if raw else None


def _read_env_file(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    if not path.is_file():
        raise ValueError(f"env file path is not a file: {path}")

    values: dict[str, str] = {}
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line.removeprefix("export ").strip()
        if "=" not in line:
            raise ValueError(f"invalid .env line {line_number}: expected KEY=VALUE")
        key, value = line.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"invalid .env line {line_number}: key is empty")
        values[key] = _clean_env_value(value)
    return values


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

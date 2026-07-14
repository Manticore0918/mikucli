from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PurePath


DEFAULT_SENSITIVE_DIRECTORY_NAMES = frozenset(
    {
        ".aws",
        ".azure",
        ".docker",
        ".gnupg",
        ".kube",
        ".ssh",
        "gcloud",
    }
)
DEFAULT_SENSITIVE_FILE_NAMES = frozenset(
    {
        ".dockercfg",
        ".netrc",
        ".npmrc",
        ".pypirc",
        "_netrc",
        "credentials",
        "id_dsa",
        "id_ecdsa",
        "id_ed25519",
        "id_rsa",
    }
)
DEFAULT_SENSITIVE_CONFIG_STEMS = frozenset({"credential", "credentials", "secret", "secrets"})
DEFAULT_SENSITIVE_CONFIG_SUFFIXES = frozenset({".cfg", ".conf", ".ini", ".json", ".toml", ".yaml", ".yml"})
DEFAULT_PRIVATE_KEY_SUFFIXES = frozenset({".jks", ".key", ".keystore", ".p12", ".pem", ".pfx"})
DEFAULT_ENV_TEMPLATE_NAMES = frozenset({".env.example", ".env.sample", ".env.template"})


@dataclass(frozen=True)
class SensitivePathMatch:
    reason: str


@dataclass(frozen=True)
class SensitivePathPolicy:
    """Classify credential-bearing paths without inspecting their contents."""

    sensitive_directory_names: frozenset[str] = DEFAULT_SENSITIVE_DIRECTORY_NAMES
    sensitive_file_names: frozenset[str] = DEFAULT_SENSITIVE_FILE_NAMES
    sensitive_config_stems: frozenset[str] = DEFAULT_SENSITIVE_CONFIG_STEMS
    sensitive_config_suffixes: frozenset[str] = DEFAULT_SENSITIVE_CONFIG_SUFFIXES
    private_key_suffixes: frozenset[str] = DEFAULT_PRIVATE_KEY_SUFFIXES
    env_template_names: frozenset[str] = DEFAULT_ENV_TEMPLATE_NAMES

    def match(self, path: str | PurePath) -> SensitivePathMatch | None:
        candidate = Path(path)
        parts = tuple(part.casefold() for part in candidate.parts if part not in {"", "."})
        if not parts:
            return None

        name = parts[-1]
        parent_parts = parts[:-1]
        if any(part in self.sensitive_directory_names for part in parent_parts):
            return SensitivePathMatch("inside a credential directory")
        if name in self.sensitive_file_names:
            return SensitivePathMatch("credential file name")
        if self._is_env_file(name):
            return SensitivePathMatch("environment file name")

        suffix = Path(name).suffix.casefold()
        stem = Path(name).stem.casefold()
        if suffix in self.private_key_suffixes:
            return SensitivePathMatch("private key or certificate container")
        if stem in self.sensitive_config_stems and suffix in self.sensitive_config_suffixes:
            return SensitivePathMatch("credential configuration file name")
        return None

    def _is_env_file(self, name: str) -> bool:
        if any(name.endswith(template_name) for template_name in self.env_template_names):
            return False
        return name == ".env" or ".env." in name or name.endswith(".env")


DEFAULT_SENSITIVE_PATH_POLICY = SensitivePathPolicy()

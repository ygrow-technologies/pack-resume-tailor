from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

DEFAULT_MODEL = "gpt-5.4-mini"


@dataclass(frozen=True)
class AppConfig:
    api_key: str
    model: str = DEFAULT_MODEL

    @classmethod
    def from_environment(cls) -> "AppConfig":
        load_dotenv_file(Path(".env"))
        api_key = os.environ.get("OPENAI_API_KEY", "").strip()
        if not api_key:
            raise ConfigurationError(
                "OPENAI_API_KEY is required. Set it before running the app."
            )

        model = os.environ.get("OPENAI_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL
        return cls(api_key=api_key, model=model)


class ConfigurationError(RuntimeError):
    """Raised when required application configuration is missing."""


def load_dotenv_file(path: Path) -> None:
    if not path.exists():
        return

    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value

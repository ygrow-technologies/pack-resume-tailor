from __future__ import annotations

from openai import OpenAI

from resume_tailor.config import AppConfig


def build_openai_client(config: AppConfig) -> OpenAI:
    return OpenAI(api_key=config.api_key)

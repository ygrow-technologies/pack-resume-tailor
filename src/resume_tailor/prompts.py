from __future__ import annotations

from pathlib import Path


DEFAULT_PROMPTS_DIR = Path("data/prompts")


def load_prompt(prompt_path: Path) -> str:
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")
    if not prompt_path.is_file():
        raise ValueError(f"Expected a prompt file path: {prompt_path}")

    prompt = prompt_path.read_text(encoding="utf-8").strip()
    if not prompt:
        raise ValueError(f"Prompt file is empty: {prompt_path}")
    return prompt


def resolve_prompt(prompt: str | None, prompt_path: Path) -> str:
    if prompt is not None:
        prompt = prompt.strip()
        if not prompt:
            raise ValueError("Prompt text cannot be empty.")
        return prompt

    return load_prompt(prompt_path)

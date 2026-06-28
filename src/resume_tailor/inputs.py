from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TailoringInput:
    resume_text: str
    job_description_text: str


def load_text_file(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    if not path.is_file():
        raise ValueError(f"Expected a file path: {path}")

    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"File is empty: {path}")
    return text


def load_tailoring_input(resume_path: Path, job_description_path: Path) -> TailoringInput:
    return TailoringInput(
        resume_text=load_text_file(resume_path),
        job_description_text=load_text_file(job_description_path),
    )

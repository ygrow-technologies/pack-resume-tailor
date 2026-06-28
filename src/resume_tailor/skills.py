from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from openai import OpenAI

from resume_tailor.pricing import ApiCallCost, calculate_text_call_cost, token_usage_from_response
from resume_tailor.prompts import DEFAULT_PROMPTS_DIR, resolve_prompt


DEFAULT_SKILL_EXTRACTION_PROMPT_PATH = (
    DEFAULT_PROMPTS_DIR / "skill-extraction-prompt.txt"
)


SKILL_EXTRACTION_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "hard_skills": {
            "type": "array",
            "items": {"type": "string"},
        },
        "soft_skills": {
            "type": "array",
            "items": {"type": "string"},
        },
    },
    "required": ["hard_skills", "soft_skills"],
}


@dataclass(frozen=True)
class ExtractedSkills:
    hard_skills: list[str]
    soft_skills: list[str]


@dataclass(frozen=True)
class SkillExtractionResult:
    skills: ExtractedSkills
    cost: ApiCallCost
    response_id: str


def extract_skills(
    client: OpenAI,
    text: str,
    model: str,
    prompt_path: Path = DEFAULT_SKILL_EXTRACTION_PROMPT_PATH,
    prompt: str | None = None,
    previous_response_id: str | None = None,
) -> SkillExtractionResult:
    prompt_text = resolve_prompt(prompt, prompt_path)
    response_args: dict[str, object] = {
        "model": model,
        "instructions": prompt_text,
        "input": text,
        "text": {
            "format": {
                "type": "json_schema",
                "name": "skill_extraction",
                "schema": SKILL_EXTRACTION_SCHEMA,
                "strict": True,
            }
        },
    }
    if previous_response_id is not None:
        response_args["previous_response_id"] = previous_response_id

    response = client.responses.create(
        **response_args,
    )
    payload = json.loads(response.output_text)
    usage = token_usage_from_response(response)
    cost = calculate_text_call_cost(model=model, usage=usage)

    return SkillExtractionResult(
        skills=ExtractedSkills(
            hard_skills=payload["hard_skills"],
            soft_skills=payload["soft_skills"],
        ),
        cost=cost,
        response_id=response.id,
    )

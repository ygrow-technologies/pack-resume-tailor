from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from openai import OpenAI

from resume_tailor.pricing import ApiCallCost, calculate_text_call_cost, token_usage_from_response
from resume_tailor.prompts import DEFAULT_PROMPTS_DIR, resolve_prompt


DEFAULT_TECHNICAL_SKILLS_GENERATION_PROMPT_PATH = (
    DEFAULT_PROMPTS_DIR / "technical-skills-generation-prompt.txt"
)


TECHNICAL_SKILLS_GENERATION_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "technicalSkills": {
            "type": "array",
            "items": {"type": "string"},
        },
    },
    "required": ["technicalSkills"],
}


@dataclass(frozen=True)
class TechnicalSkillsGenerationResult:
    technical_skills: list[str]
    cost: ApiCallCost
    response_id: str


def generate_technical_skills(
    client: OpenAI,
    resume_headline: str,
    model: str,
    prompt_path: Path = DEFAULT_TECHNICAL_SKILLS_GENERATION_PROMPT_PATH,
    prompt: str | None = None,
    previous_response_id: str | None = None,
) -> TechnicalSkillsGenerationResult:
    prompt_text = resolve_prompt(prompt, prompt_path)
    response_args: dict[str, object] = {
        "model": model,
        "instructions": prompt_text,
        "input": resume_headline,
        "text": {
            "format": {
                "type": "json_schema",
                "name": "technical_skills_generation",
                "schema": TECHNICAL_SKILLS_GENERATION_SCHEMA,
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

    return TechnicalSkillsGenerationResult(
        technical_skills=payload["technicalSkills"],
        cost=cost,
        response_id=response.id,
    )

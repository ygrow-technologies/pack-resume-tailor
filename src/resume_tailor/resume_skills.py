from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from openai import OpenAI

from resume_tailor.pricing import ApiCallCost, calculate_text_call_cost, token_usage_from_response
from resume_tailor.prompts import DEFAULT_PROMPTS_DIR, resolve_prompt


DEFAULT_RESUME_SKILLS_GENERATION_PROMPT_PATH = (
    DEFAULT_PROMPTS_DIR / "resume-skills-generation-prompt.txt"
)


RESUME_SKILLS_GENERATION_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "categorizedSkills": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "category": {"type": "string"},
                    "skills": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
                "required": ["category", "skills"],
            },
        },
    },
    "required": ["categorizedSkills"],
}


@dataclass(frozen=True)
class CategorizedSkillGroup:
    category: str
    skills: list[str]


@dataclass(frozen=True)
class ResumeSkillsGenerationResult:
    categorized_skills: list[CategorizedSkillGroup]
    cost: ApiCallCost
    response_id: str


def generate_resume_skills(
    client: OpenAI,
    extracted_skills: dict[str, object],
    technical_skills: dict[str, object],
    model: str,
    prompt_path: Path = DEFAULT_RESUME_SKILLS_GENERATION_PROMPT_PATH,
    prompt: str | None = None,
    previous_response_id: str | None = None,
) -> ResumeSkillsGenerationResult:
    prompt_text = resolve_prompt(prompt, prompt_path)
    input_payload = {
        "extractedSkills": extracted_skills,
        "technicalSkills": technical_skills,
    }
    response_args: dict[str, object] = {
        "model": model,
        "instructions": prompt_text,
        "input": json.dumps(input_payload, ensure_ascii=True),
        "text": {
            "format": {
                "type": "json_schema",
                "name": "resume_skills_generation",
                "schema": RESUME_SKILLS_GENERATION_SCHEMA,
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

    return ResumeSkillsGenerationResult(
        categorized_skills=[
            CategorizedSkillGroup(
                category=group["category"],
                skills=group["skills"],
            )
            for group in payload["categorizedSkills"]
        ],
        cost=cost,
        response_id=response.id,
    )

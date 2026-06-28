from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from openai import OpenAI

from resume_tailor.pricing import ApiCallCost, calculate_text_call_cost, token_usage_from_response
from resume_tailor.prompts import DEFAULT_PROMPTS_DIR, resolve_prompt


DEFAULT_PROFESSIONAL_SUMMARY_GENERATION_PROMPT_PATH = (
    DEFAULT_PROMPTS_DIR / "professional-summary-generation-prompt.txt"
)


PROFESSIONAL_SUMMARY_GENERATION_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "professionalSummary": {
            "type": "string",
        },
    },
    "required": ["professionalSummary"],
}


@dataclass(frozen=True)
class ProfessionalSummaryGenerationResult:
    professional_summary: str
    cost: ApiCallCost
    response_id: str


def generate_professional_summary(
    client: OpenAI,
    resume_headline: str,
    technical_skills: dict[str, object],
    company_experience: dict[str, object],
    years_of_experience: str,
    model: str,
    prompt_path: Path = DEFAULT_PROFESSIONAL_SUMMARY_GENERATION_PROMPT_PATH,
    prompt: str | None = None,
    previous_response_id: str | None = None,
) -> ProfessionalSummaryGenerationResult:
    prompt_text = resolve_prompt(prompt, prompt_path)
    input_payload = {
        "resumeHeadline": resume_headline,
        "technicalSkills": technical_skills,
        "companyExperience": company_experience,
        "yearsOfExperience": years_of_experience,
    }
    response_args: dict[str, object] = {
        "model": model,
        "instructions": prompt_text,
        "input": json.dumps(input_payload, ensure_ascii=True),
        "text": {
            "format": {
                "type": "json_schema",
                "name": "professional_summary_generation",
                "schema": PROFESSIONAL_SUMMARY_GENERATION_SCHEMA,
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

    return ProfessionalSummaryGenerationResult(
        professional_summary=payload["professionalSummary"],
        cost=cost,
        response_id=response.id,
    )

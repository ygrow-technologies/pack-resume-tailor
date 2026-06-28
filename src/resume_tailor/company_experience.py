from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from openai import OpenAI

from resume_tailor.pricing import ApiCallCost, calculate_text_call_cost, token_usage_from_response
from resume_tailor.prompts import DEFAULT_PROMPTS_DIR, resolve_prompt


DEFAULT_COMPANY_EXPERIENCE_GENERATION_PROMPT_PATH = (
    DEFAULT_PROMPTS_DIR / "company-experience-generation-prompt.txt"
)


COMPANY_EXPERIENCE_GENERATION_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "companyExperiences": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "companyName": {
                        "type": "string",
                    },
                    "workPeriod": {
                        "type": "string",
                    },
                    "companyRole": {
                        "type": "string",
                    },
                    "experienceBullets": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
                "required": [
                    "companyName",
                    "workPeriod",
                    "companyRole",
                    "experienceBullets",
                ],
            },
        },
    },
    "required": ["companyExperiences"],
}


@dataclass(frozen=True)
class CompanyExperience:
    company_name: str
    work_period: str
    company_role: str
    experience_bullets: list[str]


@dataclass(frozen=True)
class CompanyExperienceGenerationResult:
    company_experiences: list[CompanyExperience]
    cost: ApiCallCost
    response_id: str


def generate_company_experience(
    client: OpenAI,
    resume_headline: str,
    technical_skills: dict[str, object],
    product_scope: str,
    job_description_text: str,
    company_work_periods: str,
    model: str,
    prompt_path: Path = DEFAULT_COMPANY_EXPERIENCE_GENERATION_PROMPT_PATH,
    prompt: str | None = None,
    previous_response_id: str | None = None,
) -> CompanyExperienceGenerationResult:
    prompt_text = resolve_prompt(prompt, prompt_path)
    input_payload = {
        "resumeHeadline": resume_headline,
        "technicalSkills": technical_skills,
        "productScope": product_scope,
        "jobDescription": job_description_text,
        "companyWorkPeriods": company_work_periods,
    }
    response_args: dict[str, object] = {
        "model": model,
        "instructions": prompt_text,
        "input": json.dumps(input_payload, ensure_ascii=True),
        "text": {
            "format": {
                "type": "json_schema",
                "name": "company_experience_generation",
                "schema": COMPANY_EXPERIENCE_GENERATION_SCHEMA,
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

    return CompanyExperienceGenerationResult(
        company_experiences=[
            CompanyExperience(
                company_name=experience["companyName"],
                work_period=experience["workPeriod"],
                company_role=experience["companyRole"],
                experience_bullets=experience["experienceBullets"],
            )
            for experience in payload["companyExperiences"]
        ],
        cost=cost,
        response_id=response.id,
    )

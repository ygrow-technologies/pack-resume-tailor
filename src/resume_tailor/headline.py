from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from openai import OpenAI

from resume_tailor.pricing import ApiCallCost, calculate_text_call_cost, token_usage_from_response
from resume_tailor.prompts import DEFAULT_PROMPTS_DIR, resolve_prompt


DEFAULT_HEADLINE_GENERATION_PROMPT_PATH = (
    DEFAULT_PROMPTS_DIR / "headline-generation-prompt.txt"
)


HEADLINE_GENERATION_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "resumeHeadline": {
            "type": "string",
        },
    },
    "required": ["resumeHeadline"],
}


@dataclass(frozen=True)
class HeadlineGenerationResult:
    resume_headline: str
    cost: ApiCallCost
    response_id: str


def generate_headline(
    client: OpenAI,
    job_description_text: str,
    model: str,
    prompt_path: Path = DEFAULT_HEADLINE_GENERATION_PROMPT_PATH,
    prompt: str | None = None,
    previous_response_id: str | None = None,
) -> HeadlineGenerationResult:
    prompt_text = resolve_prompt(prompt, prompt_path)
    response_args: dict[str, object] = {
        "model": model,
        "instructions": prompt_text,
        "input": job_description_text,
        "text": {
            "format": {
                "type": "json_schema",
                "name": "headline_generation",
                "schema": HEADLINE_GENERATION_SCHEMA,
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

    return HeadlineGenerationResult(
        resume_headline=payload["resumeHeadline"],
        cost=cost,
        response_id=response.id,
    )

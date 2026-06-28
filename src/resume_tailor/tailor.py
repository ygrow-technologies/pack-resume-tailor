from __future__ import annotations

from openai import OpenAI

from resume_tailor.company_experience import (
    CompanyExperienceGenerationResult,
    generate_company_experience,
)
from resume_tailor.config import DEFAULT_MODEL
from resume_tailor.headline import HeadlineGenerationResult, generate_headline
from resume_tailor.professional_summary import (
    ProfessionalSummaryGenerationResult,
    generate_professional_summary,
)
from resume_tailor.resume_skills import (
    ResumeSkillsGenerationResult,
    generate_resume_skills,
)
from resume_tailor.skills import SkillExtractionResult, extract_skills
from resume_tailor.technical_skills import (
    TechnicalSkillsGenerationResult,
    generate_technical_skills,
)


class ResumeTailor:
    """Database-agnostic resume generation client.

    The host application owns persistence and prompt storage. Pass prompt text
    from your database into each method.
    """

    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_MODEL,
        client: OpenAI | None = None,
    ) -> None:
        self._client = client or OpenAI(api_key=api_key)
        self._model = model

    @property
    def model(self) -> str:
        return self._model

    def extract_skills(
        self,
        job_description: str,
        prompt: str,
        previous_response_id: str | None = None,
    ) -> SkillExtractionResult:
        return extract_skills(
            client=self._client,
            text=job_description,
            model=self._model,
            prompt=prompt,
            previous_response_id=previous_response_id,
        )

    def generate_headline(
        self,
        job_description: str,
        prompt: str,
        previous_response_id: str | None = None,
    ) -> HeadlineGenerationResult:
        return generate_headline(
            client=self._client,
            job_description_text=job_description,
            model=self._model,
            prompt=prompt,
            previous_response_id=previous_response_id,
        )

    def generate_technical_skills(
        self,
        resume_headline: str,
        prompt: str,
        previous_response_id: str | None = None,
    ) -> TechnicalSkillsGenerationResult:
        return generate_technical_skills(
            client=self._client,
            resume_headline=resume_headline,
            model=self._model,
            prompt=prompt,
            previous_response_id=previous_response_id,
        )

    def generate_resume_skills(
        self,
        extracted_skills: dict[str, object],
        technical_skills: dict[str, object],
        prompt: str,
        previous_response_id: str | None = None,
    ) -> ResumeSkillsGenerationResult:
        return generate_resume_skills(
            client=self._client,
            extracted_skills=extracted_skills,
            technical_skills=technical_skills,
            model=self._model,
            prompt=prompt,
            previous_response_id=previous_response_id,
        )

    def generate_company_experience(
        self,
        resume_headline: str,
        technical_skills: dict[str, object],
        product_scope: str,
        job_description: str,
        company_work_periods: str,
        prompt: str,
        previous_response_id: str | None = None,
    ) -> CompanyExperienceGenerationResult:
        return generate_company_experience(
            client=self._client,
            resume_headline=resume_headline,
            technical_skills=technical_skills,
            product_scope=product_scope,
            job_description_text=job_description,
            company_work_periods=company_work_periods,
            model=self._model,
            prompt=prompt,
            previous_response_id=previous_response_id,
        )

    def generate_professional_summary(
        self,
        resume_headline: str,
        technical_skills: dict[str, object],
        company_experience: dict[str, object],
        years_of_experience: str,
        prompt: str,
        previous_response_id: str | None = None,
    ) -> ProfessionalSummaryGenerationResult:
        return generate_professional_summary(
            client=self._client,
            resume_headline=resume_headline,
            technical_skills=technical_skills,
            company_experience=company_experience,
            years_of_experience=years_of_experience,
            model=self._model,
            prompt=prompt,
            previous_response_id=previous_response_id,
        )

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from openai import OpenAI

from resume_tailor.company_experience import (
    CompanyExperienceGenerationResult,
    generate_company_experience,
)
from resume_tailor.full_resume import (
    FullResumeGenerationResult,
    FullResumePromptPaths,
    generate_full_resume,
)
from resume_tailor.headline import HeadlineGenerationResult, generate_headline
from resume_tailor.inputs import TailoringInput
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


@dataclass(frozen=True)
class TailoringRequest:
    input_data: TailoringInput
    model: str


class ResumeTailoringService:
    def __init__(self, client: OpenAI) -> None:
        self._client = client

    def tailor_resume(self, request: TailoringRequest) -> str:
        _ = self._client
        _ = request
        raise NotImplementedError("Resume generation is not implemented yet.")

    def extract_skills(
        self,
        text: str,
        model: str,
        prompt_path: Path,
    ) -> SkillExtractionResult:
        return extract_skills(
            client=self._client,
            text=text,
            model=model,
            prompt_path=prompt_path,
        )

    def generate_headline(
        self,
        job_description_text: str,
        model: str,
        prompt_path: Path,
    ) -> HeadlineGenerationResult:
        return generate_headline(
            client=self._client,
            job_description_text=job_description_text,
            model=model,
            prompt_path=prompt_path,
        )

    def generate_technical_skills(
        self,
        resume_headline: str,
        model: str,
        prompt_path: Path,
    ) -> TechnicalSkillsGenerationResult:
        return generate_technical_skills(
            client=self._client,
            resume_headline=resume_headline,
            model=model,
            prompt_path=prompt_path,
        )

    def generate_resume_skills(
        self,
        extracted_skills: dict[str, object],
        technical_skills: dict[str, object],
        model: str,
        prompt_path: Path,
    ) -> ResumeSkillsGenerationResult:
        return generate_resume_skills(
            client=self._client,
            extracted_skills=extracted_skills,
            technical_skills=technical_skills,
            model=model,
            prompt_path=prompt_path,
        )

    def generate_company_experience(
        self,
        resume_headline: str,
        technical_skills: dict[str, object],
        product_scope: str,
        job_description_text: str,
        company_work_periods: str,
        model: str,
        prompt_path: Path,
    ) -> CompanyExperienceGenerationResult:
        return generate_company_experience(
            client=self._client,
            resume_headline=resume_headline,
            technical_skills=technical_skills,
            product_scope=product_scope,
            job_description_text=job_description_text,
            company_work_periods=company_work_periods,
            model=model,
            prompt_path=prompt_path,
        )

    def generate_professional_summary(
        self,
        resume_headline: str,
        technical_skills: dict[str, object],
        company_experience: dict[str, object],
        years_of_experience: str,
        model: str,
        prompt_path: Path,
    ) -> ProfessionalSummaryGenerationResult:
        return generate_professional_summary(
            client=self._client,
            resume_headline=resume_headline,
            technical_skills=technical_skills,
            company_experience=company_experience,
            years_of_experience=years_of_experience,
            model=model,
            prompt_path=prompt_path,
        )

    def generate_full_resume(
        self,
        job_description_text: str,
        product_scope: str,
        years_of_experience: str,
        company_work_periods: str,
        model: str,
        prompt_paths: FullResumePromptPaths,
        progress: Callable[[str], None] | None = None,
    ) -> FullResumeGenerationResult:
        return generate_full_resume(
            client=self._client,
            job_description_text=job_description_text,
            product_scope=product_scope,
            years_of_experience=years_of_experience,
            company_work_periods=company_work_periods,
            model=model,
            prompt_paths=prompt_paths,
            progress=progress,
        )

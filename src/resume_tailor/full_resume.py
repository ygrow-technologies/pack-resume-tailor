from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

from openai import OpenAI

from resume_tailor.company_experience import (
    DEFAULT_COMPANY_EXPERIENCE_GENERATION_PROMPT_PATH,
    CompanyExperienceGenerationResult,
    generate_company_experience,
)
from resume_tailor.headline import (
    DEFAULT_HEADLINE_GENERATION_PROMPT_PATH,
    HeadlineGenerationResult,
    generate_headline,
)
from resume_tailor.pricing import ApiCallCost
from resume_tailor.professional_summary import (
    DEFAULT_PROFESSIONAL_SUMMARY_GENERATION_PROMPT_PATH,
    ProfessionalSummaryGenerationResult,
    generate_professional_summary,
)
from resume_tailor.resume_skills import (
    DEFAULT_RESUME_SKILLS_GENERATION_PROMPT_PATH,
    ResumeSkillsGenerationResult,
    generate_resume_skills,
)
from resume_tailor.skills import (
    DEFAULT_SKILL_EXTRACTION_PROMPT_PATH,
    SkillExtractionResult,
    extract_skills,
)
from resume_tailor.technical_skills import (
    DEFAULT_TECHNICAL_SKILLS_GENERATION_PROMPT_PATH,
    TechnicalSkillsGenerationResult,
    generate_technical_skills,
)


JOB_DESCRIPTION_CONTEXT_INPUT = "Use the job description from the previous response context."


@dataclass(frozen=True)
class FullResumePromptPaths:
    skill_extraction: Path = DEFAULT_SKILL_EXTRACTION_PROMPT_PATH
    headline_generation: Path = DEFAULT_HEADLINE_GENERATION_PROMPT_PATH
    technical_skills_generation: Path = DEFAULT_TECHNICAL_SKILLS_GENERATION_PROMPT_PATH
    resume_skills_generation: Path = DEFAULT_RESUME_SKILLS_GENERATION_PROMPT_PATH
    company_experience_generation: Path = DEFAULT_COMPANY_EXPERIENCE_GENERATION_PROMPT_PATH
    professional_summary_generation: Path = (
        DEFAULT_PROFESSIONAL_SUMMARY_GENERATION_PROMPT_PATH
    )


@dataclass(frozen=True)
class FullResumeGenerationResult:
    skill_extraction: SkillExtractionResult
    headline_generation: HeadlineGenerationResult
    technical_skills_generation: TechnicalSkillsGenerationResult
    resume_skills_generation: ResumeSkillsGenerationResult
    company_experience_generation: CompanyExperienceGenerationResult
    professional_summary_generation: ProfessionalSummaryGenerationResult

    @property
    def total_cost_usd(self) -> Decimal:
        return sum(
            (cost.total_cost_usd for cost in self.costs),
            Decimal("0"),
        )

    @property
    def costs(self) -> list[ApiCallCost]:
        return [
            self.skill_extraction.cost,
            self.headline_generation.cost,
            self.technical_skills_generation.cost,
            self.resume_skills_generation.cost,
            self.company_experience_generation.cost,
            self.professional_summary_generation.cost,
        ]


def generate_full_resume(
    client: OpenAI,
    job_description_text: str,
    product_scope: str,
    years_of_experience: str,
    company_work_periods: str,
    model: str,
    prompt_paths: FullResumePromptPaths = FullResumePromptPaths(),
    progress: Callable[[str], None] | None = None,
) -> FullResumeGenerationResult:
    def log(message: str) -> None:
        if progress is not None:
            progress(message)

    log("Step 1/6: Extracting hard and soft skills from the job description.")
    skill_extraction = extract_skills(
        client=client,
        text=job_description_text,
        model=model,
        prompt_path=prompt_paths.skill_extraction,
    )
    log(f"Step 1/6 complete. Cost: ${skill_extraction.cost.total_cost_usd}")

    log("Step 2/6: Generating resume headline from the job description.")
    headline_generation = generate_headline(
        client=client,
        job_description_text=JOB_DESCRIPTION_CONTEXT_INPUT,
        model=model,
        prompt_path=prompt_paths.headline_generation,
        previous_response_id=skill_extraction.response_id,
    )
    log(f"Step 2/6 complete. Cost: ${headline_generation.cost.total_cost_usd}")

    log("Step 3/6: Generating typical technical skills from the headline.")
    technical_skills_generation = generate_technical_skills(
        client=client,
        resume_headline=headline_generation.resume_headline,
        model=model,
        prompt_path=prompt_paths.technical_skills_generation,
        previous_response_id=headline_generation.response_id,
    )
    log(
        "Step 3/6 complete. "
        f"Cost: ${technical_skills_generation.cost.total_cost_usd}"
    )

    extracted_skills_payload = {
        "hard_skills": skill_extraction.skills.hard_skills,
        "soft_skills": skill_extraction.skills.soft_skills,
    }
    technical_skills_payload = {
        "technicalSkills": technical_skills_generation.technical_skills,
    }
    log("Step 4/6: Combining extracted and typical skills into categorized skills.")
    resume_skills_generation = generate_resume_skills(
        client=client,
        extracted_skills=extracted_skills_payload,
        technical_skills=technical_skills_payload,
        model=model,
        prompt_path=prompt_paths.resume_skills_generation,
        previous_response_id=technical_skills_generation.response_id,
    )
    log(f"Step 4/6 complete. Cost: ${resume_skills_generation.cost.total_cost_usd}")

    categorized_skills_payload = {
        "categorizedSkills": [
            {
                "category": group.category,
                "skills": group.skills,
            }
            for group in resume_skills_generation.categorized_skills
        ],
    }
    log("Step 5/6: Generating company experience from products and categorized skills.")
    company_experience_generation = generate_company_experience(
        client=client,
        resume_headline=headline_generation.resume_headline,
        technical_skills=categorized_skills_payload,
        product_scope=product_scope,
        job_description_text=JOB_DESCRIPTION_CONTEXT_INPUT,
        company_work_periods=company_work_periods,
        model=model,
        prompt_path=prompt_paths.company_experience_generation,
        previous_response_id=resume_skills_generation.response_id,
    )
    log(
        "Step 5/6 complete. "
        f"Cost: ${company_experience_generation.cost.total_cost_usd}"
    )

    company_experience_payload = {
        "companyExperiences": [
            {
                "companyName": experience.company_name,
                "workPeriod": experience.work_period,
                "companyRole": experience.company_role,
                "experienceBullets": experience.experience_bullets,
            }
            for experience in company_experience_generation.company_experiences
        ],
    }
    log("Step 6/6: Generating professional summary.")
    professional_summary_generation = generate_professional_summary(
        client=client,
        resume_headline=headline_generation.resume_headline,
        technical_skills=categorized_skills_payload,
        company_experience=company_experience_payload,
        years_of_experience=years_of_experience,
        model=model,
        prompt_path=prompt_paths.professional_summary_generation,
        previous_response_id=company_experience_generation.response_id,
    )
    log(
        "Step 6/6 complete. "
        f"Cost: ${professional_summary_generation.cost.total_cost_usd}"
    )

    result = FullResumeGenerationResult(
        skill_extraction=skill_extraction,
        headline_generation=headline_generation,
        technical_skills_generation=technical_skills_generation,
        resume_skills_generation=resume_skills_generation,
        company_experience_generation=company_experience_generation,
        professional_summary_generation=professional_summary_generation,
    )
    log(f"Full resume workflow complete. Total cost: ${result.total_cost_usd}")
    return result

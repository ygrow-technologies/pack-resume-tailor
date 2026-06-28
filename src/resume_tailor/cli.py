from __future__ import annotations

import argparse
import json
from pathlib import Path

from resume_tailor.company_experience import (
    DEFAULT_COMPANY_EXPERIENCE_GENERATION_PROMPT_PATH,
)
from resume_tailor.config import AppConfig, ConfigurationError
from resume_tailor.full_resume import FullResumePromptPaths
from resume_tailor.headline import DEFAULT_HEADLINE_GENERATION_PROMPT_PATH
from resume_tailor.inputs import load_tailoring_input, load_text_file
from resume_tailor.openai_client import build_openai_client
from resume_tailor.pricing import ApiCallCost, PricingError
from resume_tailor.professional_summary import (
    DEFAULT_PROFESSIONAL_SUMMARY_GENERATION_PROMPT_PATH,
)
from resume_tailor.resume_skills import DEFAULT_RESUME_SKILLS_GENERATION_PROMPT_PATH
from resume_tailor.service import ResumeTailoringService, TailoringRequest
from resume_tailor.skills import DEFAULT_SKILL_EXTRACTION_PROMPT_PATH
from resume_tailor.technical_skills import (
    DEFAULT_TECHNICAL_SKILLS_GENERATION_PROMPT_PATH,
)


DEFAULT_SKILLS_OUTPUT_PATH = Path("output/extracted-skills.txt")
DEFAULT_HEADLINE_OUTPUT_PATH = Path("output/headline.txt")
DEFAULT_TECHNICAL_SKILLS_OUTPUT_PATH = Path("output/technical-skills.txt")
DEFAULT_RESUME_SKILLS_OUTPUT_PATH = Path("output/resume-skills.txt")
DEFAULT_COMPANY_EXPERIENCE_OUTPUT_PATH = Path("output/company-experience.txt")
DEFAULT_PROFESSIONAL_SUMMARY_OUTPUT_PATH = Path("output/professional-summary.txt")
DEFAULT_FULL_RESUME_OUTPUT_PATH = Path("output/full-resume.txt")
DEFAULT_PRODUCT_SCOPE_PATH = Path("data/products.txt")
DEFAULT_PERSONAL_INFO_PATH = Path("data/personal-info.txt")


def api_call_cost_payload(cost: ApiCallCost) -> dict[str, str | int]:
    return {
        "model": cost.model,
        "input_tokens": cost.usage.input_tokens,
        "cached_input_tokens": cost.usage.cached_input_tokens,
        "output_tokens": cost.usage.output_tokens,
        "input_cost_usd": str(cost.input_cost_usd),
        "cached_input_cost_usd": str(cost.cached_input_cost_usd),
        "output_cost_usd": str(cost.output_cost_usd),
        "total_cost_usd": str(cost.total_cost_usd),
    }


def write_json_output(output_path: Path, payload: dict[str, object]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(f"{json.dumps(payload, indent=2)}\n", encoding="utf-8")


def load_resume_headline(path: Path) -> str:
    text = load_text_file(path)
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return text

    resume_headline = payload.get("resumeHeadline")
    if not isinstance(resume_headline, str) or not resume_headline.strip():
        raise ValueError(f"Headline file does not contain resumeHeadline: {path}")
    return resume_headline.strip()


def load_json_file(path: Path) -> dict[str, object]:
    text = load_text_file(path)
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"File does not contain valid JSON: {path}") from exc

    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in file: {path}")
    return payload


def load_years_of_experience(path: Path) -> str:
    text = load_text_file(path)
    for line in text.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", maxsplit=1)
        normalized_key = key.strip().lower()
        if normalized_key in {"years of exp", "years of experience"}:
            years_of_experience = value.strip()
            if years_of_experience:
                return years_of_experience

    raise ValueError(f"Personal info file does not contain years of experience: {path}")


def load_company_work_periods(path: Path) -> str:
    text = load_text_file(path)
    for line in text.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", maxsplit=1)
        if key.strip().lower() == "period":
            company_work_periods = value.strip()
            if company_work_periods:
                return company_work_periods

    raise ValueError(f"Personal info file does not contain company periods: {path}")


def write_full_resume_component_outputs(result: object) -> None:
    write_json_output(
        DEFAULT_SKILLS_OUTPUT_PATH,
        {
            "hard_skills": result.skill_extraction.skills.hard_skills,
            "soft_skills": result.skill_extraction.skills.soft_skills,
            "cost": api_call_cost_payload(result.skill_extraction.cost),
        },
    )
    write_json_output(
        DEFAULT_HEADLINE_OUTPUT_PATH,
        {
            "resumeHeadline": result.headline_generation.resume_headline,
            "cost": api_call_cost_payload(result.headline_generation.cost),
        },
    )
    write_json_output(
        DEFAULT_TECHNICAL_SKILLS_OUTPUT_PATH,
        {
            "technicalSkills": result.technical_skills_generation.technical_skills,
            "cost": api_call_cost_payload(result.technical_skills_generation.cost),
        },
    )
    write_json_output(
        DEFAULT_RESUME_SKILLS_OUTPUT_PATH,
        {
            "categorizedSkills": [
                {
                    "category": group.category,
                    "skills": group.skills,
                }
                for group in result.resume_skills_generation.categorized_skills
            ],
            "cost": api_call_cost_payload(result.resume_skills_generation.cost),
        },
    )
    write_json_output(
        DEFAULT_COMPANY_EXPERIENCE_OUTPUT_PATH,
        {
            "companyExperiences": [
                {
                    "companyName": experience.company_name,
                    "workPeriod": experience.work_period,
                    "companyRole": experience.company_role,
                    "experienceBullets": experience.experience_bullets,
                }
                for experience in result.company_experience_generation.company_experiences
            ],
            "cost": api_call_cost_payload(result.company_experience_generation.cost),
        },
    )
    write_json_output(
        DEFAULT_PROFESSIONAL_SUMMARY_OUTPUT_PATH,
        {
            "professionalSummary": (
                result.professional_summary_generation.professional_summary
            ),
            "cost": api_call_cost_payload(result.professional_summary_generation.cost),
        },
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="resume-tailor",
        description="Prepare inputs for future OpenAI-powered resume tailoring.",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        help="Path to the base resume text file.",
    )
    parser.add_argument(
        "--job-description",
        type=Path,
        help="Path to the job description text file.",
    )
    parser.add_argument(
        "--headline",
        type=Path,
        default=DEFAULT_HEADLINE_OUTPUT_PATH,
        help="Path to a generated headline JSON or plain text file.",
    )
    parser.add_argument(
        "--extracted-skills",
        type=Path,
        default=DEFAULT_SKILLS_OUTPUT_PATH,
        help="Path to extracted skills JSON.",
    )
    parser.add_argument(
        "--technical-skills",
        type=Path,
        default=DEFAULT_TECHNICAL_SKILLS_OUTPUT_PATH,
        help="Path to generated technical skills JSON.",
    )
    parser.add_argument(
        "--product-scope",
        type=Path,
        default=DEFAULT_PRODUCT_SCOPE_PATH,
        help="Path to product scope text.",
    )
    parser.add_argument(
        "--company-experience",
        type=Path,
        default=DEFAULT_COMPANY_EXPERIENCE_OUTPUT_PATH,
        help="Path to generated company experience JSON.",
    )
    parser.add_argument(
        "--personal-info",
        type=Path,
        default=DEFAULT_PERSONAL_INFO_PATH,
        help="Path to personal info text.",
    )
    parser.add_argument(
        "--years-of-experience",
        help="Years of experience to use in generated resume content.",
    )
    action_group = parser.add_mutually_exclusive_group()
    action_group.add_argument(
        "--extract-skills",
        action="store_true",
        help="Extract hard and soft skills from the job description.",
    )
    action_group.add_argument(
        "--generate-headline",
        action="store_true",
        help="Generate a resume headline from the job description.",
    )
    action_group.add_argument(
        "--generate-technical-skills",
        action="store_true",
        help="Generate typical required technical skills from a resume headline.",
    )
    action_group.add_argument(
        "--generate-resume-skills",
        action="store_true",
        help="Generate final categorized resume skills.",
    )
    action_group.add_argument(
        "--generate-company-experience",
        action="store_true",
        help="Generate an ATS-friendly company role and experience bullets.",
    )
    action_group.add_argument(
        "--generate-professional-summary",
        action="store_true",
        help="Generate an ATS-friendly professional summary.",
    )
    action_group.add_argument(
        "--generate-full-resume",
        action="store_true",
        help="Run the full resume content generation workflow.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Path where command output should be written.",
    )
    parser.add_argument(
        "--skill-extraction-prompt",
        type=Path,
        default=DEFAULT_SKILL_EXTRACTION_PROMPT_PATH,
        help="Prompt file to use for skill extraction.",
    )
    parser.add_argument(
        "--headline-generation-prompt",
        type=Path,
        default=DEFAULT_HEADLINE_GENERATION_PROMPT_PATH,
        help="Prompt file to use for headline generation.",
    )
    parser.add_argument(
        "--technical-skills-generation-prompt",
        type=Path,
        default=DEFAULT_TECHNICAL_SKILLS_GENERATION_PROMPT_PATH,
        help="Prompt file to use for technical skills generation.",
    )
    parser.add_argument(
        "--resume-skills-generation-prompt",
        type=Path,
        default=DEFAULT_RESUME_SKILLS_GENERATION_PROMPT_PATH,
        help="Prompt file to use for final resume skills generation.",
    )
    parser.add_argument(
        "--company-experience-generation-prompt",
        type=Path,
        default=DEFAULT_COMPANY_EXPERIENCE_GENERATION_PROMPT_PATH,
        help="Prompt file to use for company experience generation.",
    )
    parser.add_argument(
        "--professional-summary-generation-prompt",
        type=Path,
        default=DEFAULT_PROFESSIONAL_SUMMARY_GENERATION_PROMPT_PATH,
        help="Prompt file to use for professional summary generation.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        config = AppConfig.from_environment()
        client = build_openai_client(config)
        service = ResumeTailoringService(client)
        if args.extract_skills:
            if args.job_description is None:
                parser.error("--job-description is required with --extract-skills.")
            job_description_text = load_text_file(args.job_description)
            result = service.extract_skills(
                text=job_description_text,
                model=config.model,
                prompt_path=args.skill_extraction_prompt,
            )
            output_path = args.output or DEFAULT_SKILLS_OUTPUT_PATH
            write_json_output(
                output_path,
                {
                    "hard_skills": result.skills.hard_skills,
                    "soft_skills": result.skills.soft_skills,
                    "cost": api_call_cost_payload(result.cost),
                },
            )
            print(f"Wrote extracted skills to {output_path}")
            print(f"Total API cost: ${result.cost.total_cost_usd}")
            return 0

        if args.generate_headline:
            if args.job_description is None:
                parser.error("--job-description is required with --generate-headline.")
            job_description_text = load_text_file(args.job_description)
            result = service.generate_headline(
                job_description_text=job_description_text,
                model=config.model,
                prompt_path=args.headline_generation_prompt,
            )
            output_path = args.output or DEFAULT_HEADLINE_OUTPUT_PATH
            write_json_output(
                output_path,
                {
                    "resumeHeadline": result.resume_headline,
                    "cost": api_call_cost_payload(result.cost),
                },
            )
            print(f"Wrote generated headline to {output_path}")
            print(f"Total API cost: ${result.cost.total_cost_usd}")
            return 0

        if args.generate_technical_skills:
            resume_headline = load_resume_headline(args.headline)
            result = service.generate_technical_skills(
                resume_headline=resume_headline,
                model=config.model,
                prompt_path=args.technical_skills_generation_prompt,
            )
            output_path = args.output or DEFAULT_TECHNICAL_SKILLS_OUTPUT_PATH
            write_json_output(
                output_path,
                {
                    "technicalSkills": result.technical_skills,
                    "cost": api_call_cost_payload(result.cost),
                },
            )
            print(f"Wrote generated technical skills to {output_path}")
            print(f"Total API cost: ${result.cost.total_cost_usd}")
            return 0

        if args.generate_resume_skills:
            extracted_skills = load_json_file(args.extracted_skills)
            technical_skills = load_json_file(args.technical_skills)
            result = service.generate_resume_skills(
                extracted_skills=extracted_skills,
                technical_skills=technical_skills,
                model=config.model,
                prompt_path=args.resume_skills_generation_prompt,
            )
            output_path = args.output or DEFAULT_RESUME_SKILLS_OUTPUT_PATH
            write_json_output(
                output_path,
                {
                    "categorizedSkills": [
                        {
                            "category": group.category,
                            "skills": group.skills,
                        }
                        for group in result.categorized_skills
                    ],
                    "cost": api_call_cost_payload(result.cost),
                },
            )
            print(f"Wrote final resume skills to {output_path}")
            print(f"Total API cost: ${result.cost.total_cost_usd}")
            return 0

        if args.generate_company_experience:
            if args.job_description is None:
                parser.error(
                    "--job-description is required with --generate-company-experience."
                )
            resume_headline = load_resume_headline(args.headline)
            technical_skills = load_json_file(args.technical_skills)
            product_scope = load_text_file(args.product_scope)
            job_description_text = load_text_file(args.job_description)
            company_work_periods = load_company_work_periods(args.personal_info)
            result = service.generate_company_experience(
                resume_headline=resume_headline,
                technical_skills=technical_skills,
                product_scope=product_scope,
                job_description_text=job_description_text,
                company_work_periods=company_work_periods,
                model=config.model,
                prompt_path=args.company_experience_generation_prompt,
            )
            output_path = args.output or DEFAULT_COMPANY_EXPERIENCE_OUTPUT_PATH
            write_json_output(
                output_path,
                {
                    "companyExperiences": [
                        {
                            "companyName": experience.company_name,
                            "workPeriod": experience.work_period,
                            "companyRole": experience.company_role,
                            "experienceBullets": experience.experience_bullets,
                        }
                        for experience in result.company_experiences
                    ],
                    "cost": api_call_cost_payload(result.cost),
                },
            )
            print(f"Wrote generated company experience to {output_path}")
            print(f"Total API cost: ${result.cost.total_cost_usd}")
            return 0

        if args.generate_professional_summary:
            resume_headline = load_resume_headline(args.headline)
            technical_skills = load_json_file(args.technical_skills)
            company_experience = load_json_file(args.company_experience)
            years_of_experience = (
                args.years_of_experience
                if args.years_of_experience is not None
                else load_years_of_experience(args.personal_info)
            )
            result = service.generate_professional_summary(
                resume_headline=resume_headline,
                technical_skills=technical_skills,
                company_experience=company_experience,
                years_of_experience=years_of_experience,
                model=config.model,
                prompt_path=args.professional_summary_generation_prompt,
            )
            output_path = args.output or DEFAULT_PROFESSIONAL_SUMMARY_OUTPUT_PATH
            write_json_output(
                output_path,
                {
                    "professionalSummary": result.professional_summary,
                    "cost": api_call_cost_payload(result.cost),
                },
            )
            print(f"Wrote generated professional summary to {output_path}")
            print(f"Total API cost: ${result.cost.total_cost_usd}")
            return 0

        if args.generate_full_resume:
            if args.job_description is None:
                parser.error("--job-description is required with --generate-full-resume.")
            job_description_text = load_text_file(args.job_description)
            product_scope = load_text_file(args.product_scope)
            years_of_experience = (
                args.years_of_experience
                if args.years_of_experience is not None
                else load_years_of_experience(args.personal_info)
            )
            company_work_periods = load_company_work_periods(args.personal_info)
            result = service.generate_full_resume(
                job_description_text=job_description_text,
                product_scope=product_scope,
                years_of_experience=years_of_experience,
                company_work_periods=company_work_periods,
                model=config.model,
                prompt_paths=FullResumePromptPaths(
                    skill_extraction=args.skill_extraction_prompt,
                    headline_generation=args.headline_generation_prompt,
                    technical_skills_generation=args.technical_skills_generation_prompt,
                    resume_skills_generation=args.resume_skills_generation_prompt,
                    company_experience_generation=args.company_experience_generation_prompt,
                    professional_summary_generation=(
                        args.professional_summary_generation_prompt
                    ),
                ),
                progress=print,
            )
            write_full_resume_component_outputs(result)
            print(
                "Updated component outputs: "
                f"{DEFAULT_SKILLS_OUTPUT_PATH}, "
                f"{DEFAULT_HEADLINE_OUTPUT_PATH}, "
                f"{DEFAULT_TECHNICAL_SKILLS_OUTPUT_PATH}, "
                f"{DEFAULT_RESUME_SKILLS_OUTPUT_PATH}, "
                f"{DEFAULT_COMPANY_EXPERIENCE_OUTPUT_PATH}, "
                f"{DEFAULT_PROFESSIONAL_SUMMARY_OUTPUT_PATH}"
            )
            output_path = args.output or DEFAULT_FULL_RESUME_OUTPUT_PATH
            write_json_output(
                output_path,
                {
                    "professionalSummary": (
                        result.professional_summary_generation.professional_summary
                    ),
                    "resumeHeadline": result.headline_generation.resume_headline,
                    "categorizedSkills": [
                        {
                            "category": group.category,
                            "skills": group.skills,
                        }
                        for group in result.resume_skills_generation.categorized_skills
                    ],
                    "companyExperiences": [
                        {
                            "companyName": experience.company_name,
                            "workPeriod": experience.work_period,
                            "companyRole": experience.company_role,
                            "experienceBullets": experience.experience_bullets,
                        }
                        for experience in (
                            result.company_experience_generation.company_experiences
                        )
                    ],
                    "intermediate": {
                        "extractedSkills": {
                            "hard_skills": (
                                result.skill_extraction.skills.hard_skills
                            ),
                            "soft_skills": (
                                result.skill_extraction.skills.soft_skills
                            ),
                        },
                        "typicalTechnicalSkills": (
                            result.technical_skills_generation.technical_skills
                        ),
                    },
                    "cost": {
                        "total_cost_usd": str(result.total_cost_usd),
                        "calls": {
                            "skillExtraction": api_call_cost_payload(
                                result.skill_extraction.cost
                            ),
                            "headlineGeneration": api_call_cost_payload(
                                result.headline_generation.cost
                            ),
                            "technicalSkillsGeneration": api_call_cost_payload(
                                result.technical_skills_generation.cost
                            ),
                            "resumeSkillsGeneration": api_call_cost_payload(
                                result.resume_skills_generation.cost
                            ),
                            "companyExperienceGeneration": api_call_cost_payload(
                                result.company_experience_generation.cost
                            ),
                            "professionalSummaryGeneration": api_call_cost_payload(
                                result.professional_summary_generation.cost
                            ),
                        },
                    },
                    "openaiSession": {
                        "usedPreviousResponseIds": True,
                        "finalResponseId": (
                            result.professional_summary_generation.response_id
                        ),
                    },
                },
            )
            print(f"Wrote generated full resume content to {output_path}")
            print(f"Total API cost: ${result.total_cost_usd}")
            return 0

        if args.resume is None:
            parser.error(
                "--resume is required unless --extract-skills or "
                "--generate-headline, --generate-technical-skills, or "
                "--generate-resume-skills, --generate-company-experience, or "
                "--generate-professional-summary, or --generate-full-resume is used."
            )

        if args.job_description is None:
            parser.error("--job-description is required for resume tailoring.")

        input_data = load_tailoring_input(args.resume, args.job_description)
        request = TailoringRequest(input_data=input_data, model=config.model)
        service.tailor_resume(request)
    except NotImplementedError as exc:
        print(f"Ready: {exc}")
        return 0
    except (ConfigurationError, FileNotFoundError, ValueError, PricingError) as exc:
        parser.error(str(exc))

    return 0

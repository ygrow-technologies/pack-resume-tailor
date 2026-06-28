# Resume Tailor

An OpenAI-powered resume tailoring package and command-line prototype.

The package exposes a database-agnostic `ResumeTailor` class. Your application provides
job descriptions, personal info, product context, and prompt text; this package handles
OpenAI calls, structured JSON parsing, response IDs, token usage, and exact cost
calculation.

## Use as a Library

Install from a local checkout:

```powershell
python -m pip install -e E:\CAI-SUB\ResumeTailor
```

Use it from another application:

```python
from resume_tailor import ResumeTailor

tailor = ResumeTailor(
    api_key=openai_api_key,
    model="gpt-5.4-mini",
)

headline = tailor.generate_headline(
    job_description=job_description_from_db,
    prompt=headline_prompt_from_db,
)

db.save_resume_step(
    run_id=run_id,
    step_name="headline",
    output={"resumeHeadline": headline.resume_headline},
    response_id=headline.response_id,
    cost_usd=str(headline.cost.total_cost_usd),
)
```

The library does not read your database. Keep database access, prompt versioning,
resume runs, retries, and saved outputs in your host application.

## Publish for Another App

For a private Git repository:

```powershell
python -m pip install git+https://github.com/your-org/resume-tailor.git
```

For a pinned version from Git:

```powershell
python -m pip install git+https://github.com/your-org/resume-tailor.git@v0.1.0
```

For PyPI or a private package index:

```powershell
python -m pip install build twine
python -m build
python -m twine upload dist/*
```

Then install it in the big app:

```powershell
python -m pip install resume-tailor
```

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e .
```

Set your OpenAI API key:

```powershell
$env:OPENAI_API_KEY = "your_api_key_here"
```

Optional model override:

```powershell
$env:OPENAI_MODEL = "gpt-5.4-mini"
```

## Run

Without installing the package, run the local PowerShell script:

```powershell
.\resume-tailor.ps1 --resume data/resume.txt --job-description data/jd-1.txt
```

After `python -m pip install -e .`, you can run the installed command:

```powershell
resume-tailor --resume data/resume.txt --job-description data/jd-1.txt
```

You can also run the package directly:

```powershell
python -m resume_tailor --resume data/resume.txt --job-description data/jd-1.txt
```

For now, the command exits after confirming the app is configured and inputs are readable.

## Extract skills

Without installing:

```powershell
.\resume-tailor.ps1 --job-description data/jd-1.txt --extract-skills
```

After `python -m pip install -e .`:

```powershell
resume-tailor --job-description data/jd-1.txt --extract-skills
```

The command writes hard skills, soft skills, token usage, and the calculated USD cost
for that exact OpenAI API response to `output/extracted-skills.txt`.

The extraction prompt lives at `data/prompts/skill-extraction-prompt.txt`, so you can
change extraction behavior without editing Python code.

To write somewhere else:

```powershell
.\resume-tailor.ps1 --job-description data/jd-1.txt --extract-skills --output output/skills.json
```

To use a different prompt file:

```powershell
.\resume-tailor.ps1 --job-description data/jd-1.txt --extract-skills --skill-extraction-prompt data/prompts/skill-extraction-prompt.txt
```

## Generate Headline

Without installing:

```powershell
.\resume-tailor.ps1 --job-description data/jd-1.txt --generate-headline
```

After `python -m pip install -e .`:

```powershell
resume-tailor --job-description data/jd-1.txt --generate-headline
```

The command writes the generated headline, token usage, and calculated USD cost for
that exact OpenAI API response to `output/headline.txt`.

The headline prompt lives at `data/prompts/headline-generation-prompt.txt`.

## Generate Technical Skills

Without installing:

```powershell
.\resume-tailor.ps1 --generate-technical-skills
```

After `python -m pip install -e .`:

```powershell
resume-tailor --generate-technical-skills
```

The command reads the headline from `output/headline.txt` by default and writes typical
required technical skills, token usage, and calculated USD cost to
`output/technical-skills.txt`.

The technical skills prompt lives at `data/prompts/technical-skills-generation-prompt.txt`.

## Generate Final Resume Skills

Without installing:

```powershell
.\resume-tailor.ps1 --generate-resume-skills
```

After `python -m pip install -e .`:

```powershell
resume-tailor --generate-resume-skills
```

The command combines `output/extracted-skills.txt` and `output/technical-skills.txt`,
then writes final categorized resume skills, token usage, and calculated USD cost to
`output/resume-skills.txt`.

The final resume skills prompt lives at
`data/prompts/resume-skills-generation-prompt.txt`.

## Generate Company Experience

Without installing:

```powershell
.\resume-tailor.ps1 --job-description data/jd-1.txt --generate-company-experience
```

After `python -m pip install -e .`:

```powershell
resume-tailor --job-description data/jd-1.txt --generate-company-experience
```

The command reads the headline from `output/headline.txt`, technical skills from
`output/technical-skills.txt`, product scope from `data/products.txt`, company work
periods from `data/personal-info.txt`, and the job description you provide. It writes
one ATS-friendly company role and experience bullet set for every `Company Name:` entry
in the product scope, including each company name, work period, token usage, and
calculated USD cost, to `output/company-experience.txt`.

The company experience prompt lives at
`data/prompts/company-experience-generation-prompt.txt`.

## Generate Professional Summary

Without installing:

```powershell
.\resume-tailor.ps1 --generate-professional-summary
```

After `python -m pip install -e .`:

```powershell
resume-tailor --generate-professional-summary
```

The command reads the headline from `output/headline.txt`, technical skills from
`output/technical-skills.txt`, company experience from `output/company-experience.txt`,
and years of experience from `data/personal-info.txt`. It writes an ATS-friendly
professional summary, including token usage and calculated USD cost, to
`output/professional-summary.txt`.

You can override years of experience directly:

```powershell
.\resume-tailor.ps1 --generate-professional-summary --years-of-experience 8
```

The professional summary prompt lives at
`data/prompts/professional-summary-generation-prompt.txt`.

## Generate Full Resume Content

Without installing:

```powershell
.\resume-tailor.ps1 --job-description data/jd-1.txt --generate-full-resume
```

After `python -m pip install -e .`:

```powershell
resume-tailor --job-description data/jd-1.txt --generate-full-resume
```

The command runs the full workflow in one action:

1. Extract hard and soft skills from the job description.
2. Generate a resume headline from the job description.
3. Generate typical technical skills from the headline.
4. Combine extracted skills and typical technical skills into final categorized skills.
5. Generate company experience from the job description, product scope, categorized skills, and headline.
6. Generate a professional summary from the headline, categorized skills, company experience, and years of experience.

It writes the full resume content, intermediate generated sections, per-step costs, total
cost, and OpenAI session metadata to `output/full-resume.txt`. The workflow chains
OpenAI response IDs between steps so later calls can reuse earlier context instead of
resending the full job description repeatedly.

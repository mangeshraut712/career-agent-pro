"""
LangChain Service Layer for CareerAgentPro
============================================
Replaces ad-hoc prompt construction with proper LangChain chains.
Demonstrates:
  - PromptTemplate + ChatOpenAI chains
  - Structured output parsing with JsonOutputParser
  - Chain composition (sequential + parallel)
  - Output parsers for reliable JSON extraction

Requirements:
  pip install langchain-core langchain-openai langchain
"""

from __future__ import annotations

import os
import json
import logging
from typing import Any

from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pydantic schemas for structured output
# ---------------------------------------------------------------------------

class JobFitAnalysis(BaseModel):
    """Structured job fit analysis output."""
    fit_score: int = Field(description="Overall fit score 0-100")
    fit_level: str = Field(description="excellent/strong/moderate/weak")
    strengths: list[str] = Field(description="Top 3-5 matching strengths")
    gaps: list[str] = Field(description="Top 3-5 gaps to address")
    recommendations: list[str] = Field(description="Actionable improvement steps")
    company_stage: str = Field(description="early_stage/growth_stage/enterprise")
    recommended_bullet_distribution: dict[str, int] = Field(
        description="Suggested bullet counts per section"
    )


class ResumeTailoring(BaseModel):
    """Structured resume tailoring output."""
    tailored_summary: str = Field(description="Rewritten professional summary")
    key_skills_to_highlight: list[str] = Field(description="Skills to emphasize")
    experience_suggestions: list[dict[str, str]] = Field(
        description="Per-experience suggestions with action and rationale"
    )
    keywords_to_add: list[str] = Field(description="Missing keywords to incorporate")
    ats_optimization_tips: list[str] = Field(description="ATS-specific improvements")


class CoverLetterDraft(BaseModel):
    """Structured cover letter output."""
    opening_hook: str = Field(description="Compelling opening paragraph")
    body: str = Field(description="2-3 paragraph body with evidence")
    closing: str = Field(description="Strong call-to-action closing")
    full_text: str = Field(description="Complete cover letter")


# ---------------------------------------------------------------------------
# LLM Factory
# ---------------------------------------------------------------------------

def _get_llm(temperature: float = 0.2, model: str | None = None) -> ChatOpenAI:
    """Create a ChatOpenAI instance configured for OpenRouter."""
    api_key = os.getenv("OPENROUTER_API_KEY", "")
    base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    model_name = model or os.getenv("LANGCHAIN_MODEL", "google/gemini-2.0-flash-exp:free")

    return ChatOpenAI(
        model=model_name,
        temperature=temperature,
        openai_api_key=api_key,
        openai_api_base=base_url,
        default_headers={
            "HTTP-Referer": os.getenv("OPENROUTER_APP_URL", "https://ai-job-helper-steel.vercel.app"),
            "X-Title": "CareerAgentPro",
        },
        max_tokens=2048,
    )


# ---------------------------------------------------------------------------
# Chain: Job Fit Analysis
# ---------------------------------------------------------------------------

def create_job_fit_chain():
    """
    LangChain chain that analyzes job fit with structured output.

    Flow: PromptTemplate → ChatOpenAI → JsonOutputParser → JobFitAnalysis
    """
    parser = JsonOutputParser(pydantic_object=JobFitAnalysis)

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """You are an expert career coach and ATS specialist.
Analyze the fit between a candidate's resume and a job description.
Be honest about gaps — don't sugarcoat. Focus on actionable advice.

{format_instructions}""",
        ),
        (
            "human",
            """## Job Description
{job_description}

## Candidate Resume
{resume_data}

Analyze the fit and return a structured assessment.""",
        ),
    ])

    llm = _get_llm(temperature=0.1)

    return prompt | llm | parser, parser


# ---------------------------------------------------------------------------
# Chain: Resume Tailoring
# ---------------------------------------------------------------------------

def create_resume_tailoring_chain():
    """
    LangChain chain that generates resume tailoring suggestions.

    Flow: PromptTemplate → ChatOpenAI → JsonOutputParser → ResumeTailoring
    """
    parser = JsonOutputParser(pydantic_object=ResumeTailoring)

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """You are an expert resume writer specializing in ATS optimization.
Tailor the resume for maximum relevance to the target job.
Focus on keyword alignment, quantifiable achievements, and ATS compatibility.

{format_instructions}""",
        ),
        (
            "human",
            """## Target Job
{job_description}

## Current Resume
{resume_data}

Provide specific tailoring recommendations.""",
        ),
    ])

    llm = _get_llm(temperature=0.2)

    return prompt | llm | parser, parser


# ---------------------------------------------------------------------------
# Chain: Cover Letter Generation
# ---------------------------------------------------------------------------

def create_cover_letter_chain():
    """
    LangChain chain that generates a cover letter.

    Flow: PromptTemplate → ChatOpenAI → JsonOutputParser → CoverLetterDraft
    """
    parser = JsonOutputParser(pydantic_object=CoverLetterDraft)

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """You are an expert cover letter writer.
Write compelling, personalized cover letters that:
1. Open with a specific hook about the company/role
2. Connect candidate experience to job requirements with evidence
3. Close with a confident call-to-action
Keep it under 400 words. No generic fluff.

{format_instructions}""",
        ),
        (
            "human",
            """## Job Details
{job_description}

## Candidate Resume
{resume_data}

## Template Style
{template_type}

Write a cover letter for this application.""",
        ),
    ])

    llm = _get_llm(temperature=0.3)

    return prompt | llm | parser, parser


# ---------------------------------------------------------------------------
# Sequential Chain: Full Analysis Pipeline
# ---------------------------------------------------------------------------

def create_analysis_pipeline():
    """
    Sequential chain: analyze job fit → tailor resume → generate cover letter.

    Demonstrates LangChain's sequential composition:
      Chain 1 (fit analysis) → Chain 2 (tailoring) → Chain 3 (cover letter)

    Each chain passes relevant outputs forward as inputs.
    """
    fit_chain, fit_parser = create_job_fit_chain()
    tailoring_chain, tailoring_parser = create_resume_tailoring_chain()
    cover_letter_chain, cl_parser = create_cover_letter_chain()

    return {
        "fit_analysis": (fit_chain, fit_parser),
        "tailoring": (tailoring_chain, tailoring_parser),
        "cover_letter": (cover_letter_chain, cl_parser),
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def analyze_job_fit(job_description: str, resume_data: dict) -> dict:
    """Run job fit analysis via LangChain chain."""
    chain, parser = create_job_fit_chain()

    try:
        result = await chain.ainvoke({
            "job_description": job_description,
            "resume_data": json.dumps(resume_data, indent=2),
            "format_instructions": parser.get_format_instructions(),
        })
        return result if isinstance(result, dict) else result.dict()
    except Exception as e:
        logger.error(f"LangChain job fit analysis failed: {e}")
        raise


async def tailor_resume(job_description: str, resume_data: dict) -> dict:
    """Run resume tailoring via LangChain chain."""
    chain, parser = create_resume_tailoring_chain()

    try:
        result = await chain.ainvoke({
            "job_description": job_description,
            "resume_data": json.dumps(resume_data, indent=2),
            "format_instructions": parser.get_format_instructions(),
        })
        return result if isinstance(result, dict) else result.dict()
    except Exception as e:
        logger.error(f"LangChain resume tailoring failed: {e}")
        raise


async def generate_cover_letter_langchain(
    job_description: str,
    resume_data: dict,
    template_type: str = "professional",
) -> dict:
    """Generate cover letter via LangChain chain."""
    chain, parser = create_cover_letter_chain()

    try:
        result = await chain.ainvoke({
            "job_description": job_description,
            "resume_data": json.dumps(resume_data, indent=2),
            "template_type": template_type,
            "format_instructions": parser.get_format_instructions(),
        })
        return result if isinstance(result, dict) else result.dict()
    except Exception as e:
        logger.error(f"LangChain cover letter generation failed: {e}")
        raise


async def run_full_pipeline(
    job_description: str,
    resume_data: dict,
    template_type: str = "professional",
) -> dict:
    """
    Run the full sequential analysis pipeline.
    Demonstrates chain composition for complex multi-step AI workflows.
    """
    chains = create_analysis_pipeline()

    resume_str = json.dumps(resume_data, indent=2)

    # Step 1: Job fit analysis
    fit_chain, fit_parser = chains["fit_analysis"]
    fit_result = await fit_chain.ainvoke({
        "job_description": job_description,
        "resume_data": resume_str,
        "format_instructions": fit_parser.get_format_instructions(),
    })

    # Step 2: Resume tailoring
    tailoring_chain, tailoring_parser = chains["tailoring"]
    tailoring_result = await tailoring_chain.ainvoke({
        "job_description": job_description,
        "resume_data": resume_str,
        "format_instructions": tailoring_parser.get_format_instructions(),
    })

    # Step 3: Cover letter
    cl_chain, cl_parser = chains["cover_letter"]
    cover_letter_result = await cl_chain.ainvoke({
        "job_description": job_description,
        "resume_data": resume_str,
        "template_type": template_type,
        "format_instructions": cl_parser.get_format_instructions(),
    })

    return {
        "fit_analysis": fit_result if isinstance(fit_result, dict) else fit_result.dict(),
        "tailoring": tailoring_result if isinstance(tailoring_result, dict) else tailoring_result.dict(),
        "cover_letter": cover_letter_result if isinstance(cover_letter_result, dict) else cover_letter_result.dict(),
        "pipeline": "langchain_sequential",
    }

"""Unit tests for ApplicationOrchestrator pipeline execution and error handling."""

import pytest
from unittest.mock import patch, AsyncMock
from services.outreach_service import OutreachCreator
from services.workflow_orchestrator import (
    ApplicationOrchestrator,
    job_context_from_description,
)


@pytest.mark.asyncio
async def test_run_full_pipeline_success():
    """Verify autonomous pipeline transitions through all 7 stages to completion."""
    job_desc = "Senior Python Backend Engineer with FastAPI, PostgreSQL, and Docker experience."
    resume_data = {
        "summary": "Experienced backend engineer building high-throughput distributed systems.",
        "skills": ["Python", "FastAPI", "Docker", "PostgreSQL"],
        "education": [{"degree": "B.S. in Computer Science", "school": "Drexel University"}],
        "experience": [
            {
                "company": "Tech Corp",
                "role": "Software Engineer",
                "bullets": [
                    "Engineered resilient microservices in Python reducing latency by 40%.",
                    "Architected event-driven data streaming pipelines handling 50k events per second.",
                ],
            }
        ],
    }
    bullet_library = [
        {
            "id": "b1",
            "text": "Engineered resilient microservices in Python reducing latency by 40%.",
            "competency": "system_design",
        }
    ]

    with patch("services.cover_letter_service.CoverLetterService.generate", new_callable=AsyncMock) as mock_cl:
        mock_cl.return_value = {
            "content": "Dear Hiring Manager,\n\nI am writing to express my enthusiastic interest in the role...",
            "template": "standard",
        }
        results = await ApplicationOrchestrator.run_full_pipeline(
            job_description=job_desc,
            resume_data=resume_data,
            bullet_library=bullet_library,
        )

    assert results["status"] == "completed"
    assert len(results["steps"]) == 7
    job_data = mock_cl.await_args.args[0]
    assert job_data["title"] not in {"excellent", "strong", "moderate", "weak"}
    assert job_data["company"] != "Target Company"
    assert job_data["description"] == job_desc
    assert "assessment" in results
    assert "selected_bullets" in results
    assert "tailored_resume" in results
    assert "resume_verification" in results
    assert "cover_letter" in results
    assert "cl_verification" in results
    assert "outreach" in results


def test_job_context_does_not_use_a_fit_tier_as_the_title():
    short = job_context_from_description("Senior Python Engineer\nBuild APIs.")
    assert short["title"] == "Senior Python Engineer"
    # The company is never known from the description alone: leave it out
    # rather than invent "the company in the job description".
    assert short["company"] == ""

    long_line = "x" * 81
    long = job_context_from_description(long_line)
    assert long["title"] == ""
    assert long["description"] == long_line

    empty = job_context_from_description("")
    assert empty["description"] == ""
    assert empty["title"] == ""


@pytest.mark.parametrize(
    "description",
    [
        # A section heading before the role name is not the role.
        "About the role\nSenior Backend Engineer\nWe are hiring.",
        "Job Description\n\nWe want a Data Scientist.",
        "Overview:\nBuild the platform.",
        # A fit tier is a score, not the role.
        "excellent\nBuild things with us.",
        # A sentence is not a title.
        "We are looking for someone great!",
    ],
)
def test_job_context_leaves_the_title_empty_when_no_line_names_the_role(description):
    assert job_context_from_description(description)["title"] == ""


def test_outreach_renders_readably_when_the_title_is_unknown():
    # The orchestrator yields an empty title; the outreach templates must not
    # write "the About the role role", "the the open role role" or
    # "the company in the job description".
    context = job_context_from_description("About the role\nSenior Backend Engineer")
    strategy = OutreachCreator.create_strategy(context, {"name": "Ada"})

    assert len(strategy.messages) > 0
    for message in strategy.messages:
        for text in (message.subject or "", message.message):
            assert "the the" not in text
            assert "role role" not in text
            assert "job description" not in text
            assert "{" not in text
            assert "\x00" not in text
    assert "I just applied for the open role at your company" in strategy.messages[0].message


def test_outreach_subjects_carry_the_real_title_not_the_placeholder():
    strategy = OutreachCreator.create_strategy(
        {"title": "Data Scientist", "company": "Acme"}, {"name": "Ada"}
    )
    subjects = [m.subject for m in strategy.messages if m.subject]
    assert subjects
    assert all("{" not in s for s in subjects)
    assert "Data Scientist Application – Ada" in subjects


@pytest.mark.asyncio
async def test_run_full_pipeline_none_and_empty_inputs():
    """Pipeline safely tolerates None/empty input data without uncaught exceptions."""
    with patch("services.cover_letter_service.CoverLetterService.generate", new_callable=AsyncMock) as mock_cl:
        mock_cl.return_value = {
            "content": "Generic cover letter content.",
            "template": "fallback",
        }
        results = await ApplicationOrchestrator.run_full_pipeline(
            job_description=None,
            resume_data=None,
            bullet_library=None,
        )

    assert results["status"] == "completed"
    assert len(results["steps"]) == 7


@pytest.mark.asyncio
async def test_run_full_pipeline_handles_exceptions():
    """Pipeline catches and records stage failures cleanly in results dict."""
    with patch("services.jd_assessor.JDAssessor.assess", side_effect=RuntimeError("Parsing error")):
        results = await ApplicationOrchestrator.run_full_pipeline(
            job_description="Broken job description",
            resume_data={},
            bullet_library=[],
        )

    assert results["status"] == "failed"
    assert "Parsing error" in results["error"]

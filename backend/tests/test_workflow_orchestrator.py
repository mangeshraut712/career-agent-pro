"""Unit tests for ApplicationOrchestrator pipeline execution and error handling."""

import pytest
from unittest.mock import patch, AsyncMock
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
    assert short["company"] == "the company in the job description"

    long_line = "x" * 81
    long = job_context_from_description(long_line)
    assert long["title"] == "the open role"
    assert long["description"] == long_line

    empty = job_context_from_description("")
    assert empty["description"] == ""
    assert empty["title"] == "the open role"


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

"""
Unit tests for CoverLetterService - 4-Paragraph Framework.
"""

import pytest
from unittest.mock import AsyncMock, patch
from services.cover_letter_service import CoverLetterService


def test_clean_content_strips_ai_prefixes_and_markdown():
    """Verify prefix removal and markdown backtick stripping."""
    raw = """Here is a cover letter:
```markdown
I am thrilled to apply for the Staff PM role at Acme Corp.
My 8 years scaling high-throughput distributed systems drives 40% efficiency.
Acme's mission to democratize workflows resonates strongly.
I look forward to discussing how my experience fits your team.
```"""
    cleaned = CoverLetterService._clean_content(raw)
    assert not cleaned.startswith("Here is a cover letter:")
    assert "```markdown" not in cleaned
    assert "```" not in cleaned
    assert "I am thrilled to apply for the Staff PM role" in cleaned


def test_clean_content_handles_none_and_empty():
    """None and empty inputs should return empty string."""
    assert CoverLetterService._clean_content(None) == ""
    assert CoverLetterService._clean_content("") == ""
    assert CoverLetterService._clean_content("   \n\t  ") == ""


def test_build_prompt_handles_none_inputs():
    """_build_prompt safely constructs prompt when arguments are None."""
    prompt = CoverLetterService._build_prompt(None, None, "minimalist", None)
    assert isinstance(prompt, str)
    assert "Job Title: Senior Product Manager" in prompt
    assert "Company: Target Company" in prompt
    assert "Candidate" in prompt


def test_build_prompt_handles_missing_fields():
    """_build_prompt handles dicts with None values."""
    job = {"title": None, "company": None, "description": None}
    resume = {"name": None, "experience": None}
    prompt = CoverLetterService._build_prompt(job, resume, "unknown_template", "Custom hook")
    assert "Custom hook" in prompt
    assert "Job Requirements:" in prompt


@pytest.mark.asyncio
async def test_generate_success_with_mocked_ai():
    """Verify async generate orchestration and structured output dictionary."""
    mock_text = """I am writing to express my strong enthusiasm for the Lead Engineer role.
Over the past 5 years, I architected distributed microservices reducing p99 latency by 45%.
Your team's commitment to reliability and scale aligns with my core technical focus.
I would welcome the opportunity to discuss how I can contribute to your infrastructure."""

    with patch("services.cover_letter_service.AIService") as mock_ai_cls:
        mock_ai_instance = AsyncMock()
        mock_ai_instance.generate_text.return_value = mock_text
        mock_ai_cls.return_value = mock_ai_instance

        job_data = {
            "title": "Lead Infrastructure Engineer",
            "company": "CloudScale",
            "description": "Seeking expert in high-throughput systems."
        }
        resume_data = {
            "name": "Jane Doe",
            "experience": [{"role": "Senior Engineer", "impact": "Reduced latency 45%"}]
        }

        res = await CoverLetterService.generate(job_data, resume_data, template_type="enterprise")
        assert res["template"] == "enterprise"
        assert res["structure"] == "Hook -> Value -> Alignment -> CTA"
        assert len(res["rules"]) == 3
        assert "CloudScale" not in res["content"] or "infrastructure" in res["content"]
        assert "Lead Engineer" in res["content"]


@pytest.mark.asyncio
async def test_generate_fallback_template_on_invalid():
    """Invalid template names default back to minimalist."""
    with patch("services.cover_letter_service.AIService") as mock_ai_cls:
        mock_ai_instance = AsyncMock()
        mock_ai_instance.generate_text.return_value = "Content"
        mock_ai_cls.return_value = mock_ai_instance

        res = await CoverLetterService.generate({}, {}, template_type="nonexistent_type")
        assert res["template"] == "minimalist"

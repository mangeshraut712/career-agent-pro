"""
Tests for LangChain Service Layer
==================================
Tests chain creation, structured output parsing, and public API functions.
"""

import json
import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from services.langchain_service import (
    JobFitAnalysis,
    ResumeTailoring,
    CoverLetterDraft,
    create_job_fit_chain,
    create_resume_tailoring_chain,
    create_cover_letter_chain,
    create_analysis_pipeline,
    analyze_job_fit,
    tailor_resume,
    generate_cover_letter_langchain,
    run_full_pipeline,
)


# ---------------------------------------------------------------------------
# Schema Tests
# ---------------------------------------------------------------------------

class TestPydanticSchemas:
    """Test that Pydantic schemas are valid and serializable."""

    def test_job_fit_analysis_schema(self):
        model = JobFitAnalysis(
            fit_score=85,
            fit_level="strong",
            strengths=["Python", "Leadership"],
            gaps=["Cloud experience"],
            recommendations=["Get AWS certified"],
            company_stage="growth_stage",
            recommended_bullet_distribution={"experience": 5, "skills": 3},
        )
        assert model.fit_score == 85
        data = model.model_dump()
        assert data["fit_level"] == "strong"
        assert len(data["strengths"]) == 2

    def test_resume_tailoring_schema(self):
        model = ResumeTailoring(
            tailored_summary="Experienced engineer...",
            key_skills_to_highlight=["Python", "FastAPI"],
            experience_suggestions=[{"section": "Job 1", "action": "Add metrics"}],
            keywords_to_add=["Kubernetes"],
            ats_optimization_tips=["Use standard job titles"],
        )
        assert len(model.keywords_to_add) == 1

    def test_cover_letter_draft_schema(self):
        model = CoverLetterDraft(
            opening_hook="I was excited to see...",
            body="My experience at X company...",
            closing="I look forward to discussing...",
            full_text="Dear Hiring Manager...",
        )
        assert "Dear Hiring Manager" in model.full_text


# ---------------------------------------------------------------------------
# Chain Creation Tests
# ---------------------------------------------------------------------------

class TestChainCreation:
    """Test that LangChain chains are created correctly."""

    def test_create_job_fit_chain(self):
        chain, parser = create_job_fit_chain()
        assert chain is not None
        assert parser is not None

    def test_create_resume_tailoring_chain(self):
        chain, parser = create_resume_tailoring_chain()
        assert chain is not None
        assert parser is not None

    def test_create_cover_letter_chain(self):
        chain, parser = create_cover_letter_chain()
        assert chain is not None
        assert parser is not None

    def test_create_analysis_pipeline(self):
        pipeline = create_analysis_pipeline()
        assert "fit_analysis" in pipeline
        assert "tailoring" in pipeline
        assert "cover_letter" in pipeline
        # Each entry should be a (chain, parser) tuple
        for key in ("fit_analysis", "tailoring", "cover_letter"):
            assert len(pipeline[key]) == 2


# ---------------------------------------------------------------------------
# Format Instructions Tests
# ---------------------------------------------------------------------------

class TestFormatInstructions:
    """Test that parsers produce valid format instructions."""

    def test_job_fit_parser_instructions(self):
        _, parser = create_job_fit_chain()
        instructions = parser.get_format_instructions()
        assert "fit_score" in instructions
        assert "json" in instructions.lower()

    def test_resume_tailoring_parser_instructions(self):
        _, parser = create_resume_tailoring_chain()
        instructions = parser.get_format_instructions()
        assert "tailored_summary" in instructions

    def test_cover_letter_parser_instructions(self):
        _, parser = create_cover_letter_chain()
        instructions = parser.get_format_instructions()
        assert "opening_hook" in instructions or "full_text" in instructions


# ---------------------------------------------------------------------------
# Public API Tests (mocked LLM)
# ---------------------------------------------------------------------------

class TestPublicAPI:
    """Test public API functions with mocked LLM responses."""

    @pytest.mark.asyncio
    async def test_analyze_job_fit(self):
        mock_result = {
            "fit_score": 80,
            "fit_level": "strong",
            "strengths": ["Python"],
            "gaps": ["Kubernetes"],
            "recommendations": ["Learn k8s"],
            "company_stage": "growth_stage",
            "recommended_bullet_distribution": {"experience": 5},
        }

        with patch("services.langchain_service.create_job_fit_chain") as mock_create:
            mock_chain = AsyncMock()
            mock_chain.ainvoke.return_value = mock_result
            mock_parser = MagicMock()
            mock_parser.get_format_instructions.return_value = "format"
            mock_create.return_value = (mock_chain, mock_parser)

            result = await analyze_job_fit("Senior Python Dev needed", {"name": "Test"})
            assert result["fit_score"] == 80
            mock_chain.ainvoke.assert_called_once()

    @pytest.mark.asyncio
    async def test_tailor_resume(self):
        mock_result = {
            "tailored_summary": "Great engineer",
            "key_skills_to_highlight": ["Python"],
            "experience_suggestions": [],
            "keywords_to_add": ["Docker"],
            "ats_optimization_tips": ["Use keywords"],
        }

        with patch("services.langchain_service.create_resume_tailoring_chain") as mock_create:
            mock_chain = AsyncMock()
            mock_chain.ainvoke.return_value = mock_result
            mock_parser = MagicMock()
            mock_parser.get_format_instructions.return_value = "format"
            mock_create.return_value = (mock_chain, mock_parser)

            result = await tailor_resume("Python Dev", {"name": "Test"})
            assert "tailored_summary" in result

    @pytest.mark.asyncio
    async def test_generate_cover_letter_langchain(self):
        mock_result = {
            "opening_hook": "Dear team",
            "body": "My experience...",
            "closing": "Best regards",
            "full_text": "Dear team...",
        }

        with patch("services.langchain_service.create_cover_letter_chain") as mock_create:
            mock_chain = AsyncMock()
            mock_chain.ainvoke.return_value = mock_result
            mock_parser = MagicMock()
            mock_parser.get_format_instructions.return_value = "format"
            mock_create.return_value = (mock_chain, mock_parser)

            result = await generate_cover_letter_langchain("Dev role", {"name": "Test"})
            assert "full_text" in result

    @pytest.mark.asyncio
    async def test_full_pipeline(self):
        mock_fit = {"fit_score": 75, "fit_level": "moderate", "strengths": [], "gaps": [], "recommendations": [], "company_stage": "enterprise", "recommended_bullet_distribution": {}}
        mock_tailor = {"tailored_summary": "", "key_skills_to_highlight": [], "experience_suggestions": [], "keywords_to_add": [], "ats_optimization_tips": []}
        mock_cl = {"opening_hook": "", "body": "", "closing": "", "full_text": ""}

        with patch("services.langchain_service.create_analysis_pipeline") as mock_create:
            fit_chain = AsyncMock()
            fit_chain.ainvoke.return_value = mock_fit
            fit_parser = MagicMock()
            fit_parser.get_format_instructions.return_value = "fmt"

            tailor_chain = AsyncMock()
            tailor_chain.ainvoke.return_value = mock_tailor
            tailor_parser = MagicMock()
            tailor_parser.get_format_instructions.return_value = "fmt"

            cl_chain = AsyncMock()
            cl_chain.ainvoke.return_value = mock_cl
            cl_parser = MagicMock()
            cl_parser.get_format_instructions.return_value = "fmt"

            mock_create.return_value = {
                "fit_analysis": (fit_chain, fit_parser),
                "tailoring": (tailor_chain, tailor_parser),
                "cover_letter": (cl_chain, cl_parser),
            }

            result = await run_full_pipeline("Job desc", {"name": "Test"})
            assert result["pipeline"] == "langchain_sequential"
            assert "fit_analysis" in result
            assert "tailoring" in result
            assert "cover_letter" in result

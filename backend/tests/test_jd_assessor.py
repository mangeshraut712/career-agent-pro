import pytest
from services.jd_assessor import JDAssessor, FitLevel, term_in_text


def test_term_in_text_matching():
    assert term_in_text("We need an experienced python engineer", "python") is True
    assert term_in_text("Collaborate with multiple engineering teams daily", "team") is True
    assert term_in_text("Must write javascript code", "java") is False
    assert term_in_text("Strong relationship building skills", "ship") is False
    assert term_in_text("", "python") is False
    assert term_in_text(None, "python") is False
    assert term_in_text("Python", None) is False


def test_analyze_skills_intelligence():
    jd = (
        "We are looking for a software architect with strong python, sql, and react skills. "
        "Experience with gcp cloud and generative ai llms is a big plus."
    )
    intel = JDAssessor.analyze_skills_intelligence(jd)
    assert "python" in intel["prioritized_tier_1"]
    assert "sql" in intel["prioritized_tier_1"]
    assert "react" in intel["prioritized_tier_1"]
    assert intel["tier_2_swaps"].get("aws") == "gcp"
    assert "ai_heavy" in intel["detected_domains"]
    assert isinstance(intel["recommendations"], list)


def test_analyze_skills_intelligence_empty_input():
    intel = JDAssessor.analyze_skills_intelligence("")
    assert intel["prioritized_tier_1"] == []
    assert intel["detected_domains"] == []

    intel_none = JDAssessor.analyze_skills_intelligence(None)
    assert intel_none["prioritized_tier_1"] == []


def test_jd_assessor_assess_strong_fit():
    jd = (
        "Remote Senior Python developer to lead cloud architecture on AWS with SQL databases. "
        "High growth team with 0-to-1 product ownership."
    )
    resume = {
        "skills": ["python", "aws", "sql", "api", "database", "lead", "product", "growth"],
        "experience": [
            {
                "role": "Lead Engineer",
                "company": "Tech Corp",
                "description": "Led team building scalable cloud APIs with python and managed SQL databases on AWS. Drove 0-to-1 product delivery and user growth."
            }
        ]
    }
    assessment = JDAssessor.assess(jd, resume)
    assert assessment.fit_score >= 85
    assert assessment.fit_level in (FitLevel.STRONG, FitLevel.EXCELLENT)
    assert len(assessment.strengths) > 0
    assert assessment.interest_level >= 8
    assert assessment.decision == "PROCEED (High Priority)"


def test_jd_assessor_assess_empty_profile():
    jd = "Software engineer"
    assessment = JDAssessor.assess(jd, {})
    assert assessment.fit_score >= 0
    assert isinstance(assessment.action_items, list)
    assert isinstance(assessment.gaps, list)

    assessment_none = JDAssessor.assess(None, None)
    assert assessment_none.fit_score == 0

"""
Unit tests for ResumeVerifier service.
"""

import pytest
from schemas import ResumeData, SixPointBullet
from services.resume_verifier import ResumeVerifier


def test_verify_resume_handles_none_gracefully():
    """verify_resume(None) must return a safe failed structure without raising."""
    result = ResumeVerifier.verify_resume(None)
    assert result["overall_pass"] is False
    assert result["overall_quality_score"] == 0
    assert result["can_export"] is False
    assert "Resume data is required" in result["errors"]
    assert result["quality_report"]["quality_level"] == "Needs Improvement"


def test_verify_resume_complete_profile():
    """Verify high-quality complete resume passes verification."""
    resume = ResumeData(
        name="Alex Morgan",
        email="alex.morgan@example.com",
        phone="555-0199",
        linkedin="https://linkedin.com/in/alexmorgan",
        summary="Accomplished Senior Product Manager with over 7 years of experience launching fintech applications.",
        experience=[
            {
                "company": "Fintech Global",
                "role": "Senior Product Manager",
                "duration": "2021-Present",
                "description": "Led team of 15 engineers delivering scalable payment APIs."
            }
        ],
        education=[
            {
                "institution": "Stanford University",
                "degree": "B.S. in Computer Science",
                "graduation_year": "2018"
            }
        ],
        skills=["Product Strategy", "API Design", "Agile", "SQL", "Data Analytics"]
    )

    result = ResumeVerifier.verify_resume(resume)
    assert result["overall_pass"] is True
    assert result["overall_quality_score"] >= 85
    assert result["can_export"] is True
    assert result["quality_report"]["quality_level"] in {"Good", "Excellent"}


def test_verify_resume_with_bullets():
    """Verify resume with valid and invalid 6-point bullets."""
    resume = ResumeData(
        name="Morgan Lee",
        email="morgan.lee@example.com",
        summary="Experienced Engineering Director with 10+ years driving distributed systems and cloud modernization.",
        experience=[
            {
                "company": "Cloud Corp",
                "role": "Director of Engineering",
                "description": "Spearheaded platform architecture and automated multi-region deployments."
            }
        ],
        education=[{"institution": "Tech Institute", "degree": "BS"}],
        skills=["Architecture", "Kubernetes", "Go", "Python", "Leadership"]
    )

    bullet = SixPointBullet(
        action="Spearheaded",
        context="global database migration for high volume core billing systems",
        method="leveraging automated phased canary deployments across regions",
        result="reducing query latency by 45% and system faults by 80%",
        impact="which unlocked 20 million dollars in annual recurring savings",
        outcome="for tier one enterprise global clients"
    )

    verif_res = ResumeVerifier.verify_resume(resume, bullets=[bullet, None])
    assert "bullet_validation" in verif_res
    assert len(verif_res["bullet_validation"]) == 1
    assert verif_res["bullet_validation"][0]["bullet_index"] == 0


def test_verify_resume_adhoc_non_dict_experience():
    """Verify non-dict experience items in custom resume structures produce warnings without throwing."""
    class CustomResume:
        name = "Jordan Reed"
        email = "jordan@example.com"
        phone = None
        linkedin = None
        summary = "Technical program manager leading infrastructure programs across multiple continents successfully."
        experience = ["Led migration at BigTech", {"company": "Acme", "role": "PM", "description": "Valid description exceeding 20 chars"}]
        education = []
        skills = ["TPM", "Agile", "Cloud"]
        projects = None

    result = ResumeVerifier.verify_resume(CustomResume())
    assert any("Invalid experience record" in w for w in result["warnings"])
    assert result["overall_quality_score"] > 0


def test_verify_resume_strict_mode():
    """Verify strict_mode enforces higher quality threshold."""
    resume = ResumeData(
        name="Taylor Swiftly",
        email="taylor@example.com",
        summary="Product designer working on enterprise collaboration systems for international teams worldwide.",
        experience=[{"company": "DesignCo", "role": "Designer", "description": "Created interactive mockups"}],
        education=[],
        skills=["Figma", "UI/UX"]
    )

    res_normal = ResumeVerifier.verify_resume(resume, strict_mode=False)
    res_strict = ResumeVerifier.verify_resume(resume, strict_mode=True)
    if res_normal["overall_quality_score"] < 85:
        assert res_strict["overall_pass"] is False


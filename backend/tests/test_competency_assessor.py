import pytest
from services.competency_assessor import CompetencyAssessor


def test_assess_job_description_basic():
    jd_text = (
        "We are looking for a Senior Product Manager to define product vision and strategy, "
        "manage the roadmap, and drive prioritization. Must collaborate with cross-functional "
        "engineering and leadership teams. Required: 5+ years of PM experience."
    )
    assessment = CompetencyAssessor.assess_job_description(jd_text)

    assert "competencies" in assessment
    assert len(assessment["competencies"]) > 0
    # Check weightages exist and sum to approximately 100%
    total_weightage = sum(c["weightage"] for c in assessment["competencies"])
    assert 99.0 <= total_weightage <= 101.0
    assert assessment["company_stage"] in ["early_stage", "growth_stage", "enterprise", "unknown"]
    assert isinstance(assessment["recommendations"], list)
    assert len(assessment["recommendations"]) > 0


def test_assess_job_description_empty():
    assessment = CompetencyAssessor.assess_job_description("")
    assert assessment["competencies"] == []
    assert assessment["company_stage"] == "unknown"
    assert assessment["recommendations"] == ["Unable to extract competencies. Review JD manually."]


def test_assess_job_description_with_requirements_and_skills():
    jd = "Software engineer"
    reqs = ["Expertise in python backend development", "Strong system design"]
    skills = ["fastapi", "cloud aws", "database sql"]

    assessment = CompetencyAssessor.assess_job_description(jd, requirements=reqs, skills=skills)
    comp_names = [c["name"] for c in assessment["competencies"]]
    assert "Technical Skills" in comp_names
    for comp in assessment["competencies"]:
        assert "weightage" in comp
        assert comp["weightage"] >= 0


def test_detect_company_stage():
    assert CompetencyAssessor._detect_company_stage("early-stage seed startup 0-1 mvp") == "early_stage"
    assert CompetencyAssessor._detect_company_stage("series b scaling scale-up rapid growth") == "growth_stage"
    assert CompetencyAssessor._detect_company_stage("fortune 500 enterprise global established public company") == "enterprise"
    assert CompetencyAssessor._detect_company_stage("ordinary local shop") == "unknown"
    assert CompetencyAssessor._detect_company_stage(
        "Looking for a PM with 10-15 years experience"
    ) == "unknown"
    assert CompetencyAssessor._detect_company_stage("0-1 product builder") == "early_stage"


def test_detect_company_stage_keeps_inflected_indicators():
    # "bootstrapped" is the only stage cue here. Whole-term matching used to
    # reject it because only "bootstrap" is listed, so this read as "unknown"
    # and the assessment skipped early-stage advice.
    stage = CompetencyAssessor._detect_company_stage(
        "Join our bootstrapped software company and own the product roadmap."
    )
    assert stage == "early_stage"

    assessment = CompetencyAssessor.assess_job_description(
        "Join our bootstrapped software company and own the product roadmap."
    )
    assert assessment["company_stage"] == "early_stage"


def test_calculate_fit_score():
    jd_text = "Python backend engineer building aws cloud architecture and sql databases."
    assessment = CompetencyAssessor.assess_job_description(jd_text)

    user_skills = ["python", "sql", "aws", "backend"]
    user_exp = [{"role": "Backend Dev", "description": "Designed cloud architecture and database schema"}]

    fit = CompetencyAssessor.calculate_fit_score(assessment, user_skills, user_exp)
    assert "overall_fit" in fit
    assert fit["overall_fit"] > 0
    assert "competency_scores" in fit
    assert "top_strengths" in fit
    assert "top_gaps" in fit


def test_calculate_fit_score_empty_profile():
    jd_text = "Technical role requiring python."
    assessment = CompetencyAssessor.assess_job_description(jd_text)

    fit = CompetencyAssessor.calculate_fit_score(assessment, None, None)
    assert fit["overall_fit"] == 0
    for cs in fit["competency_scores"]:
        assert cs["your_score"] == 0
        assert cs["gap"] == 100


def test_short_keywords_do_not_match_inside_other_words():
    jd_text = "Build guides quickly and rapidly, with the ability to combine capital plans."
    names = [c["name"] for c in CompetencyAssessor._extract_competencies(jd_text.lower())]
    assert "User Research & Design" not in names  # "ui" in "build"/"guides"
    assert "Data & Analytics" not in names  # "bi" in "ability"/"combine"
    assert "Technical Skills" not in names  # "api" in "rapidly"/"capital"


def test_whole_keywords_and_plurals_still_count():
    competencies = CompetencyAssessor._extract_competencies("own the ui and two apis for the teams")
    by_name = {c["name"]: c for c in competencies}
    assert "ui" in by_name["User Research & Design"]["keywords_found"]
    assert "api" in by_name["Technical Skills"]["keywords_found"]
    assert "team" in by_name["Leadership & Team"]["keywords_found"]

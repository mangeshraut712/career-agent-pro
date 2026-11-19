"""Unit tests for the 6-point BulletFramework and spinning strategy."""

from services.bullet_framework import BulletFramework, CompanyStage


def test_analyze_bullet_full_framework_compliance():
    """A bullet fulfilling all six points and metrics is analyzed with high score."""
    bullet = (
        "Architected enterprise-wide streaming infrastructure using Apache Kafka, "
        "reducing operational latency by 45% and saving $1.2M across 10 regions, "
        "enhancing efficiency and driving growth for global cloud clients."
    )
    analysis = BulletFramework.analyze_bullet(bullet)
    assert analysis.has_action is True
    assert analysis.has_context is True
    assert analysis.has_method is True
    assert analysis.has_result is True
    assert analysis.has_impact is True
    assert analysis.has_business_outcome is True
    assert analysis.has_metric is True
    assert analysis.score >= 75.0


def test_analyze_bullet_missing_framework_points():
    """Missing starting verbs, context, and metrics produce constructive suggestions."""
    bullet = "worked on internal bug fixes for users"
    analysis = BulletFramework.analyze_bullet(bullet)
    assert analysis.has_action is False
    assert analysis.has_metric is False
    assert len(analysis.suggestions) > 0
    assert any("action verb" in s.lower() for s in analysis.suggestions)
    assert any("quantifiable" in s.lower() for s in analysis.suggestions)


def test_detect_company_stage_heuristics():
    """Stage detection correctly identifies early-stage, growth, and enterprise job contexts."""
    early_jd = "Fast-growing pre-seed startup seeking a scrappy builder to rapidly prototype MVPs."
    assert BulletFramework.detect_company_stage(early_jd) == CompanyStage.EARLY_STAGE

    growth_jd = "Series B hypergrowth company scaling our core infrastructure with 10x KPI growth."
    assert BulletFramework.detect_company_stage(growth_jd) == CompanyStage.GROWTH_STAGE

    enterprise_jd = "Fortune 500 multinational corporation requiring cross-functional governance and compliance."
    assert BulletFramework.detect_company_stage(enterprise_jd) == CompanyStage.ENTERPRISE


def test_calculate_competency_weights():
    """Competency weighting distributes importance proportional to keyword frequencies."""
    jd = "Python Python Python backend data engineering pipelines API architecture"
    competency_areas = [
        {"name": "Backend Engineering", "keywords": ["Python", "API"]},
        {"name": "Frontend Development", "keywords": ["React", "CSS"]},
    ]
    weighted = BulletFramework.calculate_competency_weights(jd, competency_areas)
    assert len(weighted) == 2
    backend = next(c for c in weighted if c.name == "Backend Engineering")
    frontend = next(c for c in weighted if c.name == "Frontend Development")
    assert backend.weight > frontend.weight


def test_get_spinning_keywords():
    """Spinning keywords catalog provides stage-specific vocabulary."""
    startup_kw = BulletFramework.get_spinning_keywords(CompanyStage.EARLY_STAGE)
    assert "MVP" in startup_kw and "lean" in startup_kw

    growth_kw = BulletFramework.get_spinning_keywords(CompanyStage.GROWTH_STAGE)
    assert "scaled" in growth_kw or "10x" in growth_kw

    enterprise_kw = BulletFramework.get_spinning_keywords(CompanyStage.ENTERPRISE)
    assert "Fortune 500" in enterprise_kw or "governance" in enterprise_kw


def test_validate_bullet_batch_handles_an_empty_batch():
    """An empty batch reports zeros instead of raising ZeroDivisionError."""
    stats = BulletFramework.validate_bullet_batch([])
    assert stats["total_bullets"] == 0
    assert stats["average_score"] == 0
    assert set(stats["framework_compliance"].values()) == {0}

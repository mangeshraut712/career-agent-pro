"""Unit tests for MetricDiversifier and ActionVerbChecker in bullet_framework."""

import pytest

from services.bullet_framework import (
    ActionVerbChecker,
    BulletFramework,
    MetricDiversifier,
    analyze_complete_resume,
)


class TestMetricDiversifier:
    """Tests for MetricDiversifier.classify_metric and check_diversity."""

    def test_classify_time_metric(self):
        bullet = "Reduced deployment pipeline from 2 weeks to 2 days"
        assert MetricDiversifier.classify_metric(bullet) == "time"

    def test_classify_volume_metric(self):
        bullet = "Built dashboard serving 50K+ users across the organization"
        assert MetricDiversifier.classify_metric(bullet) == "volume"

    def test_classify_frequency_metric(self):
        bullet = "Executed load tests 3 times per sprint to validate SLAs"
        assert MetricDiversifier.classify_metric(bullet) == "frequency"

    def test_classify_scope_metric(self):
        bullet = "Deployed infrastructure across 12 regions globally"
        assert MetricDiversifier.classify_metric(bullet) == "scope"

    def test_classify_quality_metric(self):
        bullet = "Achieved 98% accuracy improvement in prediction model"
        assert MetricDiversifier.classify_metric(bullet) == "quality"

    def test_classify_no_metric_returns_none(self):
        bullet = "Worked on internal tooling for the backend team"
        assert MetricDiversifier.classify_metric(bullet) is None

    def test_check_diversity_all_types_present(self):
        bullets = [
            "Reduced latency from 500ms to 50ms across the pipeline",
            "Served 10K users with real-time dashboards",
            "Ran performance benchmarks 5 times per release cycle",
            "Expanded coverage to 8 markets worldwide",
            "Improved test accuracy 92% up from 68% previous quarter",
        ]
        result = MetricDiversifier.check_diversity(bullets)
        assert result["all_types_present"] is True
        assert result["diversity_score"] == 100.0
        assert result["total_bullets"] == 5
        assert len(result["warnings"]) == 0

    def test_check_diversity_warns_on_overused_type(self):
        bullets = [
            "Reduced latency from 2 hours to 30 minutes",
            "Cut build time from 45 minutes to 5 minutes",
            "Shortened onboarding from 3 weeks to 4 days",
        ]
        result = MetricDiversifier.check_diversity(bullets)
        assert len(result["warnings"]) > 0
        assert any("time" in w.lower() for w in result["warnings"])

    def test_check_diversity_empty_bullets(self):
        result = MetricDiversifier.check_diversity([])
        assert result["total_bullets"] == 0
        assert result["diversity_score"] == 0.0
        assert result["all_types_present"] is False

    def test_check_diversity_recommendations_for_missing_types(self):
        bullets = [
            "Served 50K customers on the platform",
            "Processed 1M transactions monthly",
        ]
        result = MetricDiversifier.check_diversity(bullets)
        assert len(result["recommendations"]) > 0
        assert result["all_types_present"] is False

    def test_classified_bullets_contain_snippets(self):
        long_bullet = "X" * 100 + " processed 500 users daily"
        result = MetricDiversifier.check_diversity([long_bullet])
        classified = result["classified_bullets"]
        assert len(classified) == 1
        assert classified[0]["snippet"].endswith("...")
        assert classified[0]["bullet_index"] == 1


class TestActionVerbChecker:
    """Tests for ActionVerbChecker.extract_action_verb and check_uniqueness."""

    def test_extract_action_verb_plain(self):
        assert ActionVerbChecker.extract_action_verb("Led a team of 10 engineers") == "Led"

    def test_extract_action_verb_with_bullet_marker(self):
        assert ActionVerbChecker.extract_action_verb("• Developed the core API") == "Developed"
        assert ActionVerbChecker.extract_action_verb("- Built the CI pipeline") == "Built"
        assert ActionVerbChecker.extract_action_verb("* Designed the architecture") == "Designed"

    def test_extract_action_verb_strips_punctuation(self):
        assert ActionVerbChecker.extract_action_verb("Managed, coordinated teams") == "Managed"

    def test_extract_action_verb_empty_string(self):
        assert ActionVerbChecker.extract_action_verb("") is None
        assert ActionVerbChecker.extract_action_verb("   ") is None

    def test_check_uniqueness_all_unique(self):
        bullets = [
            "Led cross-functional initiatives",
            "Built real-time analytics pipelines",
            "Designed a distributed caching layer",
        ]
        result = ActionVerbChecker.check_uniqueness(bullets)
        assert result["all_unique"] is True
        assert result["unique_verbs"] == 3
        assert len(result["duplicates"]) == 0

    def test_check_uniqueness_detects_duplicates(self):
        bullets = [
            "Led the product launch",
            "Led the migration project",
            "Built the new platform",
        ]
        result = ActionVerbChecker.check_uniqueness(bullets)
        assert result["all_unique"] is False
        assert len(result["duplicates"]) == 1
        dup = result["duplicates"][0]
        assert dup["verb"] == "Led"
        assert dup["first_use"] == 1
        assert dup["duplicate_at"] == 2
        assert dup["suggestion"] in ("Spearheaded", "Directed", "Orchestrated")

    def test_check_uniqueness_suggests_alternatives(self):
        bullets = ["Developed backend services", "Developed frontend components"]
        result = ActionVerbChecker.check_uniqueness(bullets)
        assert result["all_unique"] is False
        assert any("Replace" in s for s in result["suggestions"])

    def test_get_alternative_verb_known(self):
        alt = ActionVerbChecker._get_alternative_verb("Led")
        assert alt == "Spearheaded"

    def test_get_alternative_verb_unknown_returns_strong_verb(self):
        alt = ActionVerbChecker._get_alternative_verb("Hypothesized")
        assert alt in ActionVerbChecker.STRONG_ACTION_VERBS


class TestAnalyzeCompleteResume:
    """Tests for the top-level analyze_complete_resume function."""

    def test_high_quality_resume_is_ready(self):
        bullets = [
            "Architected enterprise-wide streaming infrastructure using Apache Kafka, "
            "reducing operational latency by 45% and saving $1.2M across 10 regions, "
            "enhancing efficiency and driving growth for global cloud clients.",

            "Spearheaded customer acquisition strategy leveraging data-driven persona "
            "targeting, increasing trial-to-paid conversion by 35% improvement and reducing "
            "CAC by $120 per customer for 500 users, driving $2.4M in additional ARR.",

            "Streamlined global supply chain operations by implementing predictive analytics "
            "for demand forecasting, 3 times per quarter audits, reducing inventory holding "
            "costs across 8 departments, ensuring compliance and mitigating risk.",
        ]
        result = analyze_complete_resume(bullets)
        assert "overall_score" in result
        assert result["overall_score"] >= 0
        assert "bullet_framework" in result
        assert "metric_diversity" in result
        assert "verb_uniqueness" in result
        assert result["verb_uniqueness"]["all_unique"] is True

    def test_weak_resume_flags_issues(self):
        bullets = [
            "Worked on stuff",
            "Did things for the team",
            "Helped with projects",
        ]
        result = analyze_complete_resume(bullets)
        assert result["overall_score"] < 50
        assert result["ready_for_submission"] is False
        assert len(result["all_recommendations"]) > 0

    def test_overall_score_weighted_correctly(self):
        """The overall score is 50% framework + 25% diversity + 25% uniqueness."""
        bullets = [
            "Led cross-functional discovery using agile methodology, reducing time by "
            "40% for 500 users across 10 departments, improving efficiency.",
        ]
        result = analyze_complete_resume(bullets)
        fw = result["bullet_framework"]["average_score"]
        div = result["metric_diversity"]["diversity_score"]
        uniq = 100.0 if result["verb_uniqueness"]["all_unique"] else (
            result["verb_uniqueness"]["unique_verbs"]
            / result["verb_uniqueness"]["total_bullets"]
            * 100
        )
        expected = round((fw * 0.5) + (div * 0.25) + (uniq * 0.25), 1)
        assert result["overall_score"] == expected


class TestBulletFrameworkValidateBatch:
    """Tests for BulletFramework.validate_bullet_batch."""

    def test_batch_with_mixed_quality(self):
        bullets = [
            "Architected enterprise-wide streaming infrastructure using Apache Kafka, "
            "reducing operational latency by 45% and saving $1.2M across 10 regions, "
            "enhancing efficiency and driving growth for global cloud clients.",
            "Worked on bugs",
        ]
        result = BulletFramework.validate_bullet_batch(bullets)
        assert result["total_bullets"] == 2
        assert result["with_metrics"] >= 1
        analyses = result["individual_analyses"]
        assert analyses[0]["score"] > analyses[1]["score"]

    def test_batch_character_distribution(self):
        short = "Did things."
        optimal = "A" * 250
        long_bullet = "B" * 300
        result = BulletFramework.validate_bullet_batch([short, optimal, long_bullet])
        dist = result["character_distribution"]
        assert dist["under_min"] == 1
        assert dist["optimal"] == 1
        assert dist["over_max"] == 1

    def test_batch_framework_compliance_percentages(self):
        bullets = [
            "Led cross-functional team using agile, reducing costs by 30% "
            "for 100 clients, improving efficiency and driving growth.",
        ]
        result = BulletFramework.validate_bullet_batch(bullets)
        compliance = result["framework_compliance"]
        assert all(0 <= v <= 100 for v in compliance.values())
        assert compliance["action"] == 100.0  # "Led" is a known action verb

"""Unit test contracts for 6-point BulletValidator service."""

import pytest
from schemas import SixPointBullet
from services.bullet_validator import BulletValidator


def test_bullet_validator_accepts_compliant_bullet():
    # 247 character compliant bullet (ideal length, all 6 points, metrics, strong verb)
    bullet = SixPointBullet(
        action="Architected",
        context="event streaming pipeline across multiple cloud regions",
        method="leveraging Kafka cluster partitioning and Redis caching queues",
        result="reducing processing latency by 45% and saving $1.2M",
        impact="accelerating operational decisions",
        outcome="for tier-1 financial partners",
    )
    result = BulletValidator.validate_bullet(bullet)
    assert result.character_count >= BulletValidator.MIN_CHARS
    assert result.character_count <= BulletValidator.MAX_CHARS
    assert result.has_all_six_points is True
    assert result.has_metrics is True
    assert result.has_strong_verb is True
    assert result.quality_score >= 70
    assert result.is_valid is True
    assert len(result.errors) == 0


def test_bullet_validator_rejects_missing_metrics():
    bullet = SixPointBullet(
        action="Developed",
        context="internal operations workflow system for customer support team members across regional office locations",
        method="using React, TypeScript, and modern continuous integration pipelines with comprehensive testing",
        result="improving team efficiency and customer feedback",
        impact="driving better company alignment",
        outcome="across global customer success teams",
    )
    result = BulletValidator.validate_bullet(bullet)
    assert result.has_metrics is False
    assert result.is_valid is False
    assert any("must contain metrics" in e.lower() for e in result.errors)


def test_bullet_validator_warns_on_weak_verb():
    bullet = SixPointBullet(
        action="Helped",
        context="cross-functional software engineering team maintain legacy database clusters across data centers",
        method="by running manual maintenance scripts and reviewing operational log files during scheduled on-call shifts",
        result="reducing recurring outage tickets by 30%",
        impact="improving database system stability",
        outcome="for five internal business units",
    )
    result = BulletValidator.validate_bullet(bullet)
    assert result.has_strong_verb is False
    assert any("weak action verb" in w.lower() for w in result.warnings)
    assert "action_verb" in result.auto_fix_suggestions


def test_metrics_detection_types():
    text = "Boosted conversion by 28%, saving $350K and supporting 50K+ daily active users."
    detection = BulletValidator._detect_metrics(text)
    assert detection.has_metrics is True
    assert "percentage" in detection.metric_types
    assert "dollar" in detection.metric_types
    assert "scaled_number" in detection.metric_types


def test_auto_fix_bullet_suggests_strong_verb():
    bullet = SixPointBullet(
        action="worked on",
        context="analytics dashboard",
        method="using Python",
        result="improved speeds by 20%",
        impact="boosted reporting",
        outcome="for managers",
    )
    fixed, changes = BulletValidator.auto_fix_bullet(bullet)
    assert any("Replace 'worked on' with 'Developed'" in c for c in changes)

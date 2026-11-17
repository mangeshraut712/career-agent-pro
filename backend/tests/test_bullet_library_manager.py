"""Unit tests for BulletLibraryManager."""
import pytest
from schemas import SixPointBullet, BulletSelectionCriteria
from services.bullet_library_manager import BulletLibraryManager


@pytest.fixture(autouse=True)
def clean_bullet_library():
    """Ensure in-memory storage is empty before and after each test."""
    BulletLibraryManager.clear_library()
    yield
    BulletLibraryManager.clear_library()


def _make_valid_bullet(**overrides):
    data = {
        "action": "Led",
        "context": "cross-functional engineering and design team",
        "method": "using Agile sprints and automated CI pipelines",
        "result": "reducing deployment failure rates by 45%",
        "impact": "accelerating delivery velocity across 8 product engineering squads",
        "outcome": "delivering core platform ahead of schedule",
        "competency": "Technical Leadership",
        "company_stage": "growth_stage",
        "tags": ["agile", "leadership"],
    }
    data.update(overrides)
    return SixPointBullet(**data)


def test_add_bullet_valid():
    bullet = _make_valid_bullet()
    res = BulletLibraryManager.add_bullet(bullet)
    assert res["success"] is True
    assert res["bullet_id"] is not None
    assert res["validation"].is_valid is True

    stored = BulletLibraryManager.get_bullet(res["bullet_id"])
    assert stored is not None
    assert stored.bullet.competency == "Technical Leadership"
    assert stored.quality_score >= 80


def test_add_bullet_none_and_invalid():
    # None bullet
    res_none = BulletLibraryManager.add_bullet(None)
    assert res_none["success"] is False
    assert "cannot be None" in res_none["message"]

    # Short bullet failing 6-point validator checks
    bad_bullet = SixPointBullet(
        action="A",
        context="B",
        method="C",
        result="D",
        impact="E",
        outcome="F",
    )
    res_bad = BulletLibraryManager.add_bullet(bad_bullet, validate=True)
    assert res_bad["success"] is False
    assert "failed validation" in res_bad["message"]


def test_add_bullet_without_validation():
    bullet = SixPointBullet(
        action="Did",
        context="some basic work",
        method="manually",
        result="got it done",
        impact="helped team",
        outcome="project completed",
    )
    res = BulletLibraryManager.add_bullet(bullet, validate=False)
    assert res["success"] is True
    stored = BulletLibraryManager.get_bullet(res["bullet_id"])
    assert stored is not None
    assert stored.quality_score == 0


def test_update_bullet():
    bullet = _make_valid_bullet()
    res = BulletLibraryManager.add_bullet(bullet)
    bullet_id = res["bullet_id"]

    # Successful update
    updated_bullet = _make_valid_bullet(
        action="Drove",
        context="cross-functional engineering and design team",
        tags=["leadership", "drove"],
    )
    update_res = BulletLibraryManager.update_bullet(bullet_id, updated_bullet)
    assert update_res["success"] is True

    stored = BulletLibraryManager.get_bullet(bullet_id)
    assert stored.bullet.action == "Drove"
    assert "drove" in stored.bullet.tags

    # Non-existing bullet
    res_missing = BulletLibraryManager.update_bullet("non-existent-id", updated_bullet)
    assert res_missing["success"] is False

    # None bullet
    res_none = BulletLibraryManager.update_bullet(bullet_id, None)
    assert res_none["success"] is False


def test_delete_bullet():
    bullet = _make_valid_bullet()
    res = BulletLibraryManager.add_bullet(bullet)
    bullet_id = res["bullet_id"]

    del_res = BulletLibraryManager.delete_bullet(bullet_id)
    assert del_res["success"] is True
    assert BulletLibraryManager.get_bullet(bullet_id) is None

    # Deleting missing bullet
    del_missing = BulletLibraryManager.delete_bullet("missing-id")
    assert del_missing["success"] is False


def test_list_bullets_filtering():
    b1 = _make_valid_bullet(competency="Product Strategy", company_stage="early_stage", tags=["pm"])
    b2 = _make_valid_bullet(competency="Data Analysis", company_stage="growth_stage", tags=["analytics"])
    b3 = _make_valid_bullet(competency="Product Strategy", company_stage="growth_stage", tags=["strategy"])

    BulletLibraryManager.add_bullet(b1)
    BulletLibraryManager.add_bullet(b2)
    BulletLibraryManager.add_bullet(b3)

    # Filter by competency
    ps_bullets = BulletLibraryManager.list_bullets(competency="Product Strategy")
    assert len(ps_bullets) == 2

    # Filter by stage
    growth_bullets = BulletLibraryManager.list_bullets(company_stage="growth_stage")
    assert len(growth_bullets) == 2

    # Filter by tags
    tag_bullets = BulletLibraryManager.list_bullets(tags=["pm"])
    assert len(tag_bullets) == 1
    assert tag_bullets[0].bullet.competency == "Product Strategy"


def test_smart_select_empty_and_populated():
    criteria = BulletSelectionCriteria(
        job_description="Looking for Technical Leadership and agile architecture expertise",
        count=2,
        company_stage="growth_stage",
    )

    # Empty library selection
    empty_res = BulletLibraryManager.smart_select(criteria)
    assert empty_res["total_selected"] == 0
    assert len(empty_res["selected_bullets"]) == 0

    # Populate library
    b1 = _make_valid_bullet(competency="Technical Leadership", company_stage="growth_stage")
    b2 = _make_valid_bullet(competency="Product Strategy", company_stage="growth_stage")
    BulletLibraryManager.add_bullet(b1)
    BulletLibraryManager.add_bullet(b2)

    pop_res = BulletLibraryManager.smart_select(criteria)
    assert pop_res["total_selected"] == 2
    # Technical leadership should score higher due to match
    assert pop_res["selected_bullets"][0]["bullet"].competency == "Technical Leadership"


def test_statistics_reporting():
    stats_empty = BulletLibraryManager.get_statistics()
    assert stats_empty["total_bullets"] == 0
    assert stats_empty["avg_quality"] == 0

    b1 = _make_valid_bullet(competency="Engineering", company_stage="growth_stage")
    b2 = _make_valid_bullet(competency="Product", company_stage="enterprise")
    BulletLibraryManager.add_bullet(b1)
    BulletLibraryManager.add_bullet(b2)

    stats = BulletLibraryManager.get_statistics()
    assert stats["total_bullets"] == 2
    assert stats["avg_quality"] > 0
    assert "Engineering" in stats["competency_distribution"]
    assert "growth_stage" in stats["stage_distribution"]

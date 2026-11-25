"""Bullet selection weights must use the same whole-term keyword rules as JD scoring."""

import pytest

from services.bullet_library import BulletLibrary
from services.jd_assessor import JDAssessor


def _technical_weight(job_description: str) -> float:
    weights = BulletLibrary._compute_competency_weights(job_description)
    return weights["technical_skills"]


def test_javascript_does_not_count_as_java():
    keywords = JDAssessor.COMPETENCY_AREAS["technical_skills"]["keywords"]
    base = JDAssessor.COMPETENCY_AREAS["technical_skills"]["weight"]
    one_mention = base * (1 / len(keywords))

    assert _technical_weight("Need JavaScript skills") == pytest.approx(one_mention)
    assert _technical_weight("Need Java skills") == pytest.approx(one_mention)


def test_ship_does_not_match_inside_relationship():
    weights = BulletLibrary._compute_competency_weights(
        "Own the customer relationship through launch week"
    )
    # "launch" is an execution term. "ship" is not inside "relationship".
    assert weights["execution"] > 0
    keywords = JDAssessor.COMPETENCY_AREAS["execution"]["keywords"]
    base = JDAssessor.COMPETENCY_AREAS["execution"]["weight"]
    assert weights["execution"] == pytest.approx(base * (1 / len(keywords)))

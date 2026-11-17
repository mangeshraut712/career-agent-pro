import pytest
from services.spinning_service import SpinningStrategy, CompanyStage


def test_spin_text_transformation():
    text = "Managed patient care and designed diagnosis action plan with 45% efficiency gain."
    result = SpinningStrategy.spin_text(text, CompanyStage.GROWTH_STAGE)
    assert result["original"] == text
    assert "patient care" not in result["spun"]
    assert "time-critical service delivery" in result["spun"]
    assert "45%" in result["preserved_metrics"]
    assert len(result["changes"]) > 0
    assert result["target_stage"] == CompanyStage.GROWTH_STAGE.value


def test_spin_text_empty_and_none():
    res_empty = SpinningStrategy.spin_text("", CompanyStage.EARLY_STAGE)
    assert res_empty["spun"] == ""
    assert res_empty["changes"] == []

    res_none = SpinningStrategy.spin_text(None, CompanyStage.ENTERPRISE)
    assert res_none["spun"] == ""
    assert res_none["changes"] == []


def test_suggest_spinning_heuristics():
    startup_jd = "Early stage seed startup building 0-1 MVP with rapid deployment."
    assert SpinningStrategy.suggest_spinning("engineer", startup_jd) == CompanyStage.EARLY_STAGE

    growth_jd = "Series B rapid scaling company optimizing ARR and conversion metrics."
    assert SpinningStrategy.suggest_spinning("engineer", growth_jd) == CompanyStage.GROWTH_STAGE

    enterprise_jd = "Fortune 500 enterprise looking for strategic alignment and compliance governance."
    assert SpinningStrategy.suggest_spinning("engineer", enterprise_jd) == CompanyStage.ENTERPRISE

    assert SpinningStrategy.suggest_spinning("engineer", None) in list(CompanyStage)


def test_get_examples_for_all_stages():
    for stage in CompanyStage:
        examples = SpinningStrategy.get_examples(stage)
        assert isinstance(examples, list)
        assert len(examples) > 0
        for ex in examples:
            assert "before" in ex
            assert "after" in ex

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


def test_spin_text_matches_whole_phrases_only():
    # "mission" is a rule, but it must not be found inside "commission".
    result = SpinningStrategy.spin_text(
        "Earned a commission on $1M in sales", CompanyStage.EARLY_STAGE
    )
    assert result["spun"] == "Earned a commission on $1M in sales"
    assert result["changes"] == []

    # ...while a whole-word occurrence is translated, with capitalisation kept.
    result = SpinningStrategy.spin_text("Led the Mission", CompanyStage.EARLY_STAGE)
    assert result["spun"] == "Led the Project"
    assert [c["original"] for c in result["changes"]] == ["Mission"]


@pytest.mark.parametrize(
    "text, preserved",
    [
        ("Planned tactical exercises", "tactical exercises"),
        ("Served under command of", "under command of"),
    ],
)
def test_spin_text_does_not_change_the_nature_of_the_work(text, preserved):
    # This module adapts language "without fabrication". These used to become
    # "strategic exercises" and "leadership", each claiming a different kind of
    # work than the one described.
    result = SpinningStrategy.spin_text(text, CompanyStage.ENTERPRISE)
    assert result["changes"] == []
    assert result["spun"] == text


def test_spin_text_preserves_factual_qualifiers_while_translating_around_them():
    # "hospice" names the kind of care given; translating it ("response care")
    # would misstate the work. The surrounding jargon is still translated.
    result = SpinningStrategy.spin_text(
        "Led hospice care for families", CompanyStage.ENTERPRISE
    )
    assert result["spun"] == "Led hospice care for populations"
    assert [c["original"] for c in result["changes"]] == ["families"]


def test_spin_text_rewrite_does_not_depend_on_stage():
    # The rewrite is stage-neutral on purpose; only the guidance differs. The
    # response used to describe a shared rewrite as stage-specific.
    text = "Led hospice care for families"
    early = SpinningStrategy.spin_text(text, CompanyStage.EARLY_STAGE)
    enterprise = SpinningStrategy.spin_text(text, CompanyStage.ENTERPRISE)

    assert early["spun"] == enterprise["spun"]
    assert early["explanation"] != enterprise["explanation"]
    for change in early["changes"] + enterprise["changes"]:
        assert "early_stage" not in change["reason"]
        assert "enterprise" not in change["reason"]


def test_spin_text_rejects_unknown_stage():
    with pytest.raises(ValueError):
        SpinningStrategy.spin_text("text", "not_a_stage")

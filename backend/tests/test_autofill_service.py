"""Unit tests for AutofillService ATS mapping and JS script generation."""

import pytest
from services.autofill_service import (
    AutofillService,
    _as_text,
    _optional_yes_no,
    json_for_script,
)


def test_optional_yes_no_mapping():
    """Verify boolean and string parsing for ATS yes/no questions."""
    assert _optional_yes_no(True) == "Yes"
    assert _optional_yes_no(False) == "No"
    assert _optional_yes_no("yes") == "Yes"
    assert _optional_yes_no("  YES  ") == "Yes"
    assert _optional_yes_no("true") == "Yes"
    assert _optional_yes_no("y") == "Yes"
    assert _optional_yes_no("no") == "No"
    assert _optional_yes_no("FALSE") == "No"
    assert _optional_yes_no("n") == "No"
    assert _optional_yes_no(None) is None
    assert _optional_yes_no("unknown") is None
    assert _optional_yes_no(123) is None


def test_as_text_coercion_and_limits():
    """Verify text coercion for strings, numbers, booleans, and limits."""
    assert _as_text("developer") == "developer"
    assert _as_text(42) == "42"
    assert _as_text(3.14) == "3.14"
    assert _as_text(True) == "True"
    assert _as_text(None) is None
    assert _as_text({"nested": "dict"}) is None
    # Test length limit
    long_str = "a" * 600
    truncated = _as_text(long_str, limit=50)
    assert len(truncated) == 50


def test_json_for_script_escaping():
    """Ensure HTML tag characters and line separators cannot close script blocks."""
    payload = {
        "tag": "</script><script>alert('xss')</script>",
        "unicode": "line1\u2028line2\u2029line3",
        "amp": "Rock & Roll",
    }
    encoded = json_for_script(payload)
    assert "<" not in encoded
    assert ">" not in encoded
    assert "\\u003c" in encoded
    assert "\\u003e" in encoded
    assert "\\u0026" in encoded
    assert "\\u2028" in encoded
    assert "\\u2029" in encoded


def test_get_field_mappings_known_and_fallback():
    """Verify platform mappings for all supported ATS platforms and fallback."""
    for platform in ["google", "greenhouse", "lever", "indeed", "linkedin"]:
        mapping = AutofillService.get_field_mappings(platform)
        assert isinstance(mapping, dict)
        assert "email" in mapping or "full_name" in mapping

    # Fallback to google mapping for unknown platform
    fallback = AutofillService.get_field_mappings("workday_custom")
    google = AutofillService.get_field_mappings("google")
    assert fallback == google

    # None and empty string safely fall back to google
    assert AutofillService.get_field_mappings(None) == google
    assert AutofillService.get_field_mappings("") == google


def test_generate_autofill_script_standard():
    """Generate script with standard profile fields and check JS structure."""
    resume_data = {
        "name": "Jane Doe",
        "email": "jane.doe@example.com",
        "phone": "+1-555-0100",
        "linkedin": "https://linkedin.com/in/janedoe",
        "website": "https://janedoe.dev",
        "location": "San Francisco, CA",
        "is_authorized_us": True,
        "requires_sponsorship": False,
        "gender": "Female",
        "race": "Decline to state",
        "disability": "No",
        "is_veteran": "No",
        "sexual_orientation": "Decline to state",
        "custom_questions": {
            "Years of experience": "5",
            "Willing to relocate": "Yes",
        },
    }

    script = AutofillService.generate_autofill_script(resume_data, platform="greenhouse")
    assert isinstance(script, str)
    assert "function fillField(selectors, value)" in script
    assert "jane.doe@example.com" in script
    assert "Jane" in script
    assert "Doe" in script
    assert "Enhanced AI Autofill attempt completed" in script


def test_generate_autofill_script_none_and_invalid_data():
    """Script generation handles None or non-dict input safely."""
    script_none = AutofillService.generate_autofill_script(None, platform="lever")
    assert isinstance(script_none, str)
    assert "function fillField(selectors, value)" in script_none

    script_invalid = AutofillService.generate_autofill_script("not a dict", platform=None)
    assert isinstance(script_invalid, str)
    assert "Enhanced AI Autofill attempt completed" in script_invalid

def test_greenhouse_linkedin_is_matched_by_label_not_by_position():
    """A profile's LinkedIn URL must only reach a field that is LinkedIn's.

    The mapping used to fall back to the id of the first custom answer on any
    Greenhouse form, so on a form without a LinkedIn question the URL was typed
    into whichever question happened to be first, such as "How did you hear
    about us?".
    """
    resume_data = {
        "name": "Jane Doe",
        "email": "jane.doe@example.com",
        "linkedin": "https://linkedin.com/in/janedoe",
    }

    script = AutofillService.generate_autofill_script(resume_data, platform="greenhouse")

    assert "job_application_answers_attributes_0_text_value" not in script
    # The label-based selector is still generated, so a form that does ask for
    # LinkedIn (by aria-label or label text) is still filled.
    assert '"linkedin"' in script

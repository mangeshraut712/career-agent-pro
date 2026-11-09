"""Regression tests for the CodeQL py/polynomial-redos alerts.

The title clean-up in JobService and the summary metric check in the
verification service used patterns that retried a whitespace or digit run
from every position, so a long run took quadratic time. These tests pin the
results on normal input and bound the time on long runs.
"""

import time

from services.job_service import clean_linkedin_title
from services.verification_service import _has_summary_metric


def _elapsed(fn, *args):
    start = time.perf_counter()
    fn(*args)
    return time.perf_counter() - start


def test_clean_linkedin_title_keeps_existing_results():
    assert clean_linkedin_title("Acme hiring Senior Engineer in Pune, India | LinkedIn") == "Senior Engineer"
    assert clean_linkedin_title("Data Scientist | LinkedIn") == "Data Scientist"
    assert clean_linkedin_title("Acme Corp is HIRING  Backend Developer") == "Backend Developer"
    assert clean_linkedin_title("Product Manager in Remote") == "Product Manager"
    assert clean_linkedin_title("Staff Engineer") == "Staff Engineer"
    assert clean_linkedin_title("Engineer in training") == "Engineer in training"


def test_clean_linkedin_title_is_linear_on_long_whitespace():
    title = " " * 50_000 + "x"
    # The old patterns took several seconds here; the new ones take milliseconds.
    assert _elapsed(clean_linkedin_title, title) < 0.5
    assert _elapsed(clean_linkedin_title, " in A" + "\tin\tA" * 20_000 + "!") < 0.5


def test_summary_metric_pattern_keeps_existing_matches():
    for text in ("Grew revenue 40%", "Saved $1,200", "Led a 5k user beta", "Raised 3 M"):
        assert _has_summary_metric(text), text
    for text in ("Product manager with strong skills", "Shipped v2"):
        assert not _has_summary_metric(text), text


def test_summary_metric_pattern_is_linear_on_long_digit_runs():
    assert _elapsed(_has_summary_metric, "9" * 50_000 + "x") < 0.5

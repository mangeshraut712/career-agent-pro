"""Job source labels come from the URL host, not a substring of the whole URL."""

from services.job_service import JobService


def _source(url: str) -> str:
    service = JobService.__new__(JobService)
    return service._detect_job_source(url)


def test_boards_greenhouse_host_is_greenhouse():
    assert _source("https://boards.greenhouse.io/acme/jobs/123") == "greenhouse"
    assert _source("https://www.boards.greenhouse.io/acme/jobs/123") == "greenhouse"
    assert _source("https://boards.greenhouse.io:443/acme/jobs/123") == "greenhouse"
    assert _source("https://job-boards.greenhouse.io/acme/jobs/1") == "greenhouse"


def test_greenhouse_lookalikes_stay_other():
    assert _source("https://notgreenhouse.io/jobs/1") == "other"
    assert _source("https://evil.example/redirect?next=https://boards.greenhouse.io/acme") == "other"
    assert _source("https://boards.greenhouse.io.evil.com/acme") == "other"


def test_other_boards_still_resolve_from_the_host():
    assert _source("https://www.linkedin.com/jobs/view/1") == "linkedin"
    assert _source("https://jobs.lever.co/acme/123") == "lever"
    assert _source("https://angel.co/company/acme/jobs/1") == "wellfound"

"""Regression tests for the quality, security, and correctness fixes."""
import io
import socket
from datetime import datetime, timezone
from urllib.parse import urlparse

import pytest
from fastapi.testclient import TestClient
from fpdf import FPDF
import jwt

from auth_utils import ACCESS_TOKEN_EXPIRE_MINUTES, create_access_token
from main import app
from middleware import rate_limiter
from schemas import BulletSelectionRequest, ResumeData
from security_utils import (
    ResumeUploadError,
    docs_enabled,
    read_limited_upload,
    resolve_cors_origins,
    safe_download_name,
)
from services.ai_service import AIService
from services.autofill_service import AutofillService
from services.export_service import ExportService, escape_latex, pdf_text
from services.job_service import JobService
from services.resume_parser import ResumeParser

client = TestClient(app)

SAMPLE_RESUME = {
    "name": "José Álvarez",
    "email": "jose@example.com",
    "phone": "555-0100",
    "summary": r"Grew revenue 50% & reduced cost with \input{secret}",
    "experience": [
        {
            "role": "Engineer",
            "company": "Acme",
            "duration": "2020-2024",
            "description": "Shipped the API",
        }
    ],
    "education": [
        {"institution": "Drexel", "degree": "MS", "graduation_year": "2024"}
    ],
    "skills": ["Python", "C++"],
}


def _resume_model() -> ResumeData:
    return ResumeData(**SAMPLE_RESUME)


class TestSecretsAndAuth:
    def test_short_secret_is_rejected(self, monkeypatch):
        monkeypatch.setenv("SECRET_KEY", "short-secret")
        with pytest.raises(RuntimeError):
            create_access_token({"sub": "user"})

    def test_missing_secret_is_rejected(self, monkeypatch):
        monkeypatch.delenv("SECRET_KEY", raising=False)
        with pytest.raises(RuntimeError):
            create_access_token({"sub": "user"})

    def test_default_lifetime_is_thirty_minutes(self, monkeypatch):
        monkeypatch.setenv("SECRET_KEY", "k" * 32)
        token = create_access_token({"sub": "user"})
        payload = jwt.decode(token, "k" * 32, algorithms=["HS256"])
        remaining = payload["exp"] - datetime.now(timezone.utc).timestamp()
        assert ACCESS_TOKEN_EXPIRE_MINUTES * 60 - 5 <= remaining <= ACCESS_TOKEN_EXPIRE_MINUTES * 60 + 5

    def test_health_does_not_expose_key_material(self):
        response = client.get("/health")
        assert response.status_code == 200
        body = response.text
        assert "api_key_length" not in body
        assert "api_key_present" not in body
        assert "api_key_valid" not in body
        assert "sk-" not in body

    def test_debug_false_string_does_not_enable_docs(self, monkeypatch):
        monkeypatch.setenv("DEBUG", "false")
        assert docs_enabled() is False
        monkeypatch.setenv("DEBUG", "0")
        assert docs_enabled() is False
        monkeypatch.setenv("DEBUG", "true")
        assert docs_enabled() is True


class TestCorsAndHeaders:
    def test_wildcard_origin_is_ignored(self):
        origins = resolve_cors_origins("*,https://app.example")
        assert origins == ["https://app.example"]
        assert "*" not in origins

    def test_empty_config_uses_explicit_defaults(self):
        origins = resolve_cors_origins("  ")
        assert "http://localhost:3000" in origins
        assert "*" not in origins

    def test_browser_origin_allowlist(self):
        allowed = client.get("/health", headers={"Origin": "http://localhost:3000"})
        assert allowed.headers.get("access-control-allow-origin") == "http://localhost:3000"

        blocked = client.get("/health", headers={"Origin": "https://evil.example"})
        assert blocked.headers.get("access-control-allow-origin") != "https://evil.example"
        assert blocked.headers.get("access-control-allow-origin") != "*"

    def test_security_headers_present(self):
        response = client.get("/")
        assert response.headers["x-content-type-options"] == "nosniff"
        assert response.headers["x-frame-options"] == "DENY"
        assert "referrer-policy" in response.headers

    def test_rate_limit_blocks_excess_posts(self):
        original = rate_limiter.requests_per_minute
        rate_limiter.requests.clear()
        rate_limiter.requests_per_minute = 1
        try:
            first = client.post(
                "/detect-company-stage",
                json={"job_description": "A growth stage startup building APIs"},
            )
            second = client.post(
                "/detect-company-stage",
                json={"job_description": "An enterprise company with compliance needs"},
            )
            assert first.status_code == 200
            assert second.status_code == 429
            health = client.get("/health")
            assert health.status_code == 200
        finally:
            rate_limiter.requests_per_minute = original
            rate_limiter.requests.clear()


class TestUploads:
    def test_rejects_unknown_extension(self):
        response = client.post(
            "/parse-resume",
            files={"file": ("payload.exe", b"not a resume", "application/octet-stream")},
        )
        assert response.status_code == 400
        assert "Unsupported file type" in response.text
        assert "Traceback" not in response.text

    def test_rejects_legacy_doc(self):
        response = client.post(
            "/parse-resume",
            files={"file": ("resume.doc", b"binary", "application/msword")},
        )
        assert response.status_code == 400
        assert ".doc" in response.text

    def test_rejects_empty_file(self):
        response = client.post(
            "/parse-resume",
            files={"file": ("resume.txt", b"", "text/plain")},
        )
        assert response.status_code == 400

    def test_rejects_oversize_upload(self, monkeypatch):
        monkeypatch.setattr("security_utils.MAX_UPLOAD_BYTES", 8)
        response = client.post(
            "/parse-resume",
            files={"file": ("resume.txt", b"123456789", "text/plain")},
        )
        assert response.status_code == 413
        assert "10MB" in response.text

    def test_text_resume_parses(self, monkeypatch):
        monkeypatch.setattr("main.ai_service.is_configured", False)
        resume = b"Ada Lovelace\nada@example.com\nBuilt analytical engines.\n"
        response = client.post(
            "/parse-resume",
            files={"file": ("resume.txt", resume, "text/plain")},
        )
        assert response.status_code == 200
        assert "ada@example.com" in response.text.lower() or "Ada" in response.text

    def test_stream_endpoint_is_live(self, monkeypatch):
        monkeypatch.setattr("main.ai_service.is_configured", False)
        resume = b"Ada Lovelace\nada@example.com\nMathematician\n"
        response = client.post(
            "/parse-resume-stream",
            files={"file": ("resume.txt", resume, "text/plain")},
        )
        assert response.status_code == 200
        assert "text/event-stream" in response.headers["content-type"]
        assert "completed" in response.text

    def test_pdf_page_limit(self):
        pdf = FPDF()
        pdf.set_font("Helvetica", size=12)
        for _ in range(31):
            pdf.add_page()
            pdf.cell(0, 10, "Page of resume text for the parser limit.")
        payload = bytes(pdf.output())
        with pytest.raises(ResumeUploadError, match="30 page"):
            ResumeParser.extract_text(payload, "long.pdf")

    @pytest.mark.asyncio
    async def test_read_limited_upload_counts_chunks(self):
        class FakeUpload:
            def __init__(self):
                self._data = io.BytesIO(b"abcdefghij")

            async def read(self, size):
                return self._data.read(size)

        with pytest.raises(ResumeUploadError):
            await read_limited_upload(FakeUpload(), max_bytes=4)


class TestExportSafety:
    def test_latex_escape_is_single_pass(self):
        escaped = escape_latex(r"50% & growth \input{secret} x^2")
        assert r"50\%" in escaped
        assert r"\&" in escaped
        assert r"\textbackslash{}input\{secret\}" in escaped
        assert r"\textasciicircum{}2" in escaped
        without_commands = escaped.replace(r"\%", "").replace(r"\&", "")
        assert "%" not in without_commands
        assert "&" not in without_commands

    def test_latex_phone_uses_a_real_newline(self):
        latex = ExportService.to_latex(_resume_model())
        assert "\\phone[mobile]{555-0100}\n" in latex
        assert r"\phone[mobile]{555-0100}\n" not in latex
        assert r"\input" not in latex

    def test_pdf_text_keeps_unicode_when_font_supports_it(self):
        assert pdf_text("José 🚀", True) == "José 🚀"
        assert "🚀" not in pdf_text("José 🚀", False)

    def test_pdf_export_accepts_unicode_names(self, tmp_path):
        output = tmp_path / "resume.pdf"
        ExportService.to_pdf(_resume_model(), str(output))
        data = output.read_bytes()
        assert data.startswith(b"%PDF")
        assert len(data) > 500

    def test_download_name_strips_header_injection(self):
        assert safe_download_name("Jane\r\nSet-Cookie: evil", "pdf") == "Jane_Set-Cookie_evil_resume.pdf"
        assert "\r" not in safe_download_name("a/../../b", "docx")
        assert safe_download_name("a/../../b", "docx").endswith("_resume.docx")

    def test_pdf_response_sanitizes_filename_and_deletes_temp(self, monkeypatch):
        removed = []
        original = __import__("os").remove

        def spy(path):
            removed.append(path)
            original(path)

        monkeypatch.setattr("main.os.remove", spy)
        payload = dict(SAMPLE_RESUME)
        payload["name"] = "Jane\r\nSet-Cookie: evil"
        response = client.post("/export/pdf", json=payload)
        assert response.status_code == 200
        assert response.content.startswith(b"%PDF")
        disposition = response.headers["content-disposition"]
        assert "\r" not in disposition
        assert "\n" not in disposition
        assert removed
        assert all(not __import__("os").path.exists(path) for path in removed)


class TestJobUrlSafety:
    def setup_method(self):
        self.service = JobService(ai_service=None)

    def test_allowlist_uses_hostname_not_netloc(self):
        assert self.service._is_allowed_url("https://www.linkedin.com:443/jobs/view/1")
        assert self.service._is_allowed_url("https://jobs.lever.co/acme/123")
        assert not self.service._is_allowed_url("https://www.linkedin.com:8443/jobs")
        assert not self.service._is_allowed_url("https://user:pass@www.linkedin.com/jobs")
        assert not self.service._is_allowed_url("https://169.254.169.254/latest/meta-data")
        assert not self.service._is_allowed_url("https://evil.example/jobs")
        assert not self.service._is_allowed_url("https://notlinkedin.com/jobs")
        assert not self.service._is_allowed_url("https://linkedin.com.evil.example/jobs")

    @pytest.mark.asyncio
    async def test_private_resolution_is_not_fetched(self, monkeypatch):
        calls = []

        def fake_getaddrinfo(host, port, *args, **kwargs):
            return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("169.254.169.254", 0))]

        class FakeClient:
            def __init__(self, *args, **kwargs):
                calls.append(kwargs)

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                return None

            async def get(self, url, headers=None):
                calls.append(url)
                raise AssertionError("private address must not be requested")

        monkeypatch.setattr("services.job_service.socket.getaddrinfo", fake_getaddrinfo)
        monkeypatch.setattr("services.job_service.httpx.AsyncClient", FakeClient)
        result = await self.service._fetch_with_httpx("https://www.linkedin.com/jobs/view/1")
        assert result is None
        assert calls == []

    @pytest.mark.asyncio
    async def test_public_resolution_disables_redirects(self, monkeypatch):
        seen = {}

        def fake_getaddrinfo(host, port, *args, **kwargs):
            return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("8.8.8.8", 0))]

        class FakeResponse:
            text = "<html><title>Engineer</title></html>"

            def raise_for_status(self):
                return None

        class FakeClient:
            def __init__(self, *args, **kwargs):
                seen["init"] = kwargs

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                return None

            async def get(self, url, headers=None):
                seen["url"] = url
                return FakeResponse()

        monkeypatch.setattr("services.job_service.socket.getaddrinfo", fake_getaddrinfo)
        monkeypatch.setattr("services.job_service.httpx.AsyncClient", FakeClient)
        text = await self.service._fetch_with_httpx("https://www.linkedin.com/jobs/view/9")
        assert text
        assert seen["init"]["follow_redirects"] is False
        assert seen["url"].startswith("https://www.linkedin.com/")

    @pytest.mark.asyncio
    async def test_redirect_hops_are_revalidated(self, monkeypatch):
        requested = []
        mode = {"blocked": True}

        def fake_getaddrinfo(host, port, *args, **kwargs):
            return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("8.8.8.8", 0))]

        class FakeResponse:
            def __init__(self, status_code, text, location=None):
                self.status_code = status_code
                self.text = text
                self.headers = {"location": location} if location else {}

        class FakeClient:
            def __init__(self, *args, **kwargs):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                return None

            async def get(self, url, headers=None):
                requested.append(url)
                if mode["blocked"]:
                    return FakeResponse(302, "<html>redirect</html>", "https://evil.example/steal")
                if url == "https://www.linkedin.com/jobs/view/9":
                    return FakeResponse(302, "redirect", "https://jobs.lever.co/acme/1")
                return FakeResponse(200, "<html><title>Engineer</title>Build APIs</html>")

        monkeypatch.setattr("services.job_service.socket.getaddrinfo", fake_getaddrinfo)
        monkeypatch.setattr("services.job_service.httpx.AsyncClient", FakeClient)
        blocked = await self.service._fetch_with_httpx("https://www.linkedin.com/jobs/view/9")
        assert blocked is None
        assert requested == ["https://www.linkedin.com/jobs/view/9"]

        requested.clear()
        mode["blocked"] = False
        followed = await self.service._fetch_with_httpx("https://www.linkedin.com/jobs/view/9")
        assert followed and "Engineer" in followed
        assert requested == [
            "https://www.linkedin.com/jobs/view/9",
            "https://jobs.lever.co/acme/1",
        ]

    @pytest.mark.asyncio
    async def test_extract_does_not_call_trafilatura_fetch(self, monkeypatch):
        def explode(url):
            raise AssertionError(f"trafilatura.fetch_url was called for {url}")

        async def short_fetch(url, user_agent=None):
            return "too short"

        monkeypatch.setattr("services.job_service.trafilatura.fetch_url", explode)
        monkeypatch.setattr(self.service, "_fetch_with_httpx", short_fetch)
        result = await self.service.extract_from_url("https://www.linkedin.com/jobs/view/9")
        parsed = urlparse(result.url)
        assert parsed.scheme == "https"
        assert parsed.hostname == "www.linkedin.com"
        assert parsed.path == "/jobs/view/9"


class TestAutofillAndAiErrors:
    def test_indeed_mapping_and_script_escaping(self):
        script = AutofillService.generate_autofill_script(
            {
                "name": '</script><script>alert(1)</script>',
                "email": "ada@example.com",
                "custom_questions": {"Work auth": "Yes"},
            },
            "indeed",
        )
        assert "applicant.firstName" in script
        assert "</script>" not in script
        assert "\\u003c/script\\u003e" in script

    def test_linkedin_mapping_is_not_google_fallback(self):
        mapping = AutofillService.get_field_mappings("linkedin")
        assert "emailAddress" in mapping["email"]

    def test_omitted_demographics_are_not_invented(self):
        resume = ResumeData(**SAMPLE_RESUME)
        assert resume.gender is None
        assert resume.race is None
        assert resume.sexual_orientation is None
        assert resume.is_authorized_us is None
        assert resume.requires_sponsorship is None

        script = AutofillService.generate_autofill_script(
            {"name": "Ada Lovelace", "email": "ada@example.com"},
            "google",
        )
        assert '"is_authorized_us": null' in script
        assert '"requires_sponsorship": null' in script
        assert '"gender": null' in script
        assert "Philadelphia" not in script
        assert "Heterosexual" not in script
        assert '"is_authorized_us": "Yes"' not in script

        explicit = AutofillService.generate_autofill_script(
            {"name": "Ada Lovelace", "is_authorized_us": False, "requires_sponsorship": True},
            "google",
        )
        assert '"is_authorized_us": "No"' in explicit
        assert '"requires_sponsorship": "Yes"' in explicit

    @pytest.mark.asyncio
    async def test_completion_does_not_return_exception_text(self):
        service = AIService()
        service.is_configured = True

        class Completions:
            async def create(self, **kwargs):
                raise RuntimeError("sk-or-v1-SECRETLEAK")

        service.client = type("Client", (), {})()
        service.client.chat = type("Chat", (), {"completions": Completions()})()
        result = await service.get_completion("hello")
        assert "SECRETLEAK" not in result
        assert "API request failed" in result

    @pytest.mark.asyncio
    async def test_stream_skips_empty_choices_and_hides_errors(self):
        service = AIService()
        service.is_configured = True

        class Delta:
            content = "hello"

        class Choice:
            delta = Delta()

        class Chunk:
            def __init__(self, choices):
                self.choices = choices

        class Stream:
            def __aiter__(self):
                self._items = iter([Chunk([]), Chunk([Choice()])])
                return self

            async def __anext__(self):
                try:
                    return next(self._items)
                except StopIteration as exc:
                    raise StopAsyncIteration from exc

        class Completions:
            async def create(self, **kwargs):
                return Stream()

        service.client = type("Client", (), {})()
        service.client.chat = type("Chat", (), {"completions": Completions()})()
        chunks = [chunk async for chunk in service.stream_completion("hello")]
        assert chunks == ["hello"]

        class Failing:
            async def create(self, **kwargs):
                raise RuntimeError("sk-or-v1-SECRETLEAK")

        service.client.chat = type("Chat", (), {"completions": Failing()})()
        failed = [chunk async for chunk in service.stream_completion("hello")]
        assert "SECRETLEAK" not in "".join(failed)


class TestApiErrorHygiene:
    def test_langchain_failure_is_generic(self, monkeypatch):
        async def boom(*args, **kwargs):
            raise RuntimeError("sk-or-v1-SECRETLEAK")

        monkeypatch.setattr("main.langchain_analyze_job_fit", boom)
        response = client.post(
            "/langchain/analyze-fit",
            json={"job_description": "Senior engineer role building APIs"},
        )
        assert response.status_code == 500
        assert "SECRETLEAK" not in response.text
        assert "Traceback" not in response.text

    def test_bullet_batch_is_capped(self):
        response = client.post("/analyze-bullets", json={"bullets": ["did a thing"] * 101})
        assert response.status_code == 400
        assert "100" in response.text

    def test_stream_template_is_allowlisted(self):
        response = client.get(
            "/api/stream/cover-letter",
            params={
                "job_description": "We need a senior Python engineer with five years of experience",
                "template_type": "ignore-previous-instructions",
            },
        )
        assert response.status_code == 400

    def test_stream_message_length_is_capped(self):
        response = client.get("/api/stream/chat", params={"message": "a" * 4001})
        assert response.status_code == 422

    def test_non_string_bullets_are_rejected(self):
        response = client.post("/analyze-bullets", json={"bullets": [{"text": "not a string"}]})
        assert response.status_code == 400
        assert "string" in response.text

    def test_job_description_length_is_capped(self):
        response = client.post(
            "/assess-job-fit",
            json={"job_description": "python " * 5000, "resume_data": {}},
        )
        assert response.status_code == 400
        assert "20000" in response.text

    def test_bullet_selection_count_is_bounded(self):
        response = client.post(
            "/bullet-library/select-for-job",
            json={"job_description": "Python engineer", "bullets": [{"text": "Built APIs"}], "count": 0},
        )
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_oversized_content_length_is_rejected(self):
        from middleware import RateLimitMiddleware

        async def app(scope, receive, send):
            raise AssertionError("oversized request must not reach the app")

        middleware = RateLimitMiddleware(app)
        messages = []

        async def receive():
            return {"type": "http.request", "body": b"", "more_body": False}

        async def send(message):
            messages.append(message)

        await middleware(
            {
                "type": "http",
                "path": "/enhance-resume",
                "method": "POST",
                "headers": [(b"content-length", b"99999999")],
                "client": ("203.0.113.8", 1234),
            },
            receive,
            send,
        )
        assert messages[0]["status"] == 413

    def test_resume_selection_model_rejects_huge_batches(self):
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            BulletSelectionRequest(job_description="role", bullets=[{"text": "Built APIs"}] * 101, count=13)

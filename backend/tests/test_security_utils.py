import pytest
from security_utils import (
    body_size_status,
    docs_enabled,
    resolve_cors_origins,
    resume_extension,
    safe_download_name,
    validate_resume_filename,
    ResumeUploadError,
    DEFAULT_CORS_ORIGINS,
)


def test_docs_enabled(monkeypatch):
    monkeypatch.setenv("DEBUG", "1")
    assert docs_enabled() is True

    monkeypatch.setenv("DEBUG", "true")
    assert docs_enabled() is True

    monkeypatch.setenv("DEBUG", "yes")
    assert docs_enabled() is True

    monkeypatch.setenv("DEBUG", "on")
    assert docs_enabled() is True

    monkeypatch.setenv("DEBUG", "0")
    assert docs_enabled() is False

    monkeypatch.setenv("DEBUG", "false")
    assert docs_enabled() is False

    monkeypatch.setenv("DEBUG", "")
    assert docs_enabled() is False


def test_resolve_cors_origins():
    # Empty or None defaults
    assert resolve_cors_origins("") == list(DEFAULT_CORS_ORIGINS)
    assert resolve_cors_origins("   ") == list(DEFAULT_CORS_ORIGINS)

    # Wildcard is strictly rejected
    assert resolve_cors_origins("*") == list(DEFAULT_CORS_ORIGINS)

    # Allowed origins preserved and normalized
    origins = resolve_cors_origins("https://app.example.com/, http://localhost:8000, *")
    assert origins == ["https://app.example.com", "http://localhost:8000"]


def test_resume_extension():
    assert resume_extension("resume.pdf") == ".pdf"
    assert resume_extension("RESUME.DOCX") == ".docx"
    assert resume_extension("my.complex.name.txt") == ".txt"
    assert resume_extension("no_extension") == ""
    assert resume_extension("") == ""


def test_validate_resume_filename():
    assert validate_resume_filename("candidate.pdf") == ".pdf"
    assert validate_resume_filename("candidate.docx") == ".docx"
    assert validate_resume_filename("candidate.txt") == ".txt"

    # Legacy .doc error message
    with pytest.raises(ResumeUploadError) as exc_doc:
        validate_resume_filename("candidate.doc")
    assert "Legacy .doc files are not supported" in str(exc_doc.value)

    # Unsupported format error message
    with pytest.raises(ResumeUploadError) as exc_exe:
        validate_resume_filename("malicious.exe")
    assert "Unsupported file type" in str(exc_exe.value)


def test_body_size_status():
    # Valid sizes
    assert body_size_status("1024") is None
    assert body_size_status(None) is None
    assert body_size_status("") is None

    # Malformed non-integer length -> 400
    assert body_size_status("abc") == 400
    assert body_size_status("12.5") == 400

    # Negative length -> 413
    assert body_size_status("-1") == 413

    # Exceeding limit -> 413
    assert body_size_status("15000000", limit=10_000_000) == 413


def test_safe_download_name():
    # Normal case
    assert safe_download_name("Jane Doe", "pdf") == "Jane_Doe_resume.pdf"
    assert safe_download_name("John-Smith_Dev", ".docx") == "John-Smith_Dev_resume.docx"

    # Special characters and header injection attempts sanitized
    assert safe_download_name('Jane\r\nHeader: inject; filename="bad.exe"', "txt") == "Jane_Header_inject_filename_bad_exe_resume.txt"

    # Unsupported extension falls back to .bin
    assert safe_download_name("Profile", "sh") == "Profile_resume.bin"

    # Empty name defaults to resume
    assert safe_download_name("", "pdf") == "resume_resume.pdf"

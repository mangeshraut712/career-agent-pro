"""Unit tests for backend utility functions."""

import pytest
from utils import (
    extract_domain,
    sanitize_filename,
    truncate_text,
    validate_email,
    validate_phone,
    validate_url,
)


def test_validate_email_contracts():
    """Verify email format validation across valid and invalid inputs."""
    assert validate_email("user@example.com") is True
    assert validate_email("candidate.john+tech@sub.domain.co") is True
    assert validate_email("invalid-email") is False
    assert validate_email("@no-user.com") is False
    assert validate_email("user@") is False
    assert validate_email("") is False
    assert validate_email(None) is False


def test_validate_phone_contracts():
    """Verify phone validation adheres to 10-15 digit standards."""
    assert validate_phone("+1 (555) 123-4567") is True
    assert validate_phone("+91 98765 43210") is True
    assert validate_phone("555-0199") is False  # too short (< 10)
    assert validate_phone("1" * 20) is False  # too long (> 15)
    assert validate_phone("not-a-number") is False
    assert validate_phone("") is False
    assert validate_phone(None) is False


def test_sanitize_filename_prevents_traversal_and_null_bytes():
    """Verify directory traversal components and dangerous characters are stripped."""
    assert sanitize_filename("../../etc/passwd") == "passwd"
    assert sanitize_filename("..\\..\\windows\\win.ini") == "win.ini"
    assert sanitize_filename("safe_document.pdf") == "safe_document.pdf"
    assert sanitize_filename("file<with>bad:chars|?.txt") == "filewithbadchars.txt"
    assert sanitize_filename("null\x00byte.png") == "nullbyte.png"
    assert sanitize_filename("") == "file"
    assert sanitize_filename(".") == "file"
    assert sanitize_filename("..") == "file"
    assert sanitize_filename(None) == "file"


def test_validate_url_contracts():
    """Verify URL validation requires http or https scheme and host."""
    assert validate_url("https://careers.google.com/jobs") is True
    assert validate_url("http://localhost:8000/api") is True
    assert validate_url("ftp://ftp.example.com/file") is False
    assert validate_url("javascript:alert(1)") is False
    assert validate_url("not a url") is False
    assert validate_url("") is False
    assert validate_url(None) is False


def test_extract_domain_with_ports_and_edge_cases():
    """Verify extract_domain correctly handles URLs with ports, paths, and www stripping."""
    assert extract_domain("https://example.com:8080/path") == "example.com"
    assert extract_domain("https://www.google.com:443/search") == "google.com"
    assert extract_domain("not-a-url") is None
    assert extract_domain("") is None
    assert extract_domain(None) is None

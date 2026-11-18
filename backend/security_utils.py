"""Request validation and response-safety helpers.

These helpers keep user-controlled values out of headers, shell-like
filenames, and unbounded uploads. Error strings returned to clients are
fixed messages so exception text cannot leak secrets.
"""
from __future__ import annotations

import os
import unicodedata
from urllib.parse import urlsplit

from fastapi import UploadFile

MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_REQUEST_BYTES = 12 * 1024 * 1024
MAX_PDF_PAGES = 30
MAX_EXTRACTED_CHARS = 500_000
MAX_BULLETS = 100
MAX_TEXT_CHARS = 20_000
MAX_BULLET_CHARS = 2_000
ALLOWED_RESUME_EXTENSIONS = frozenset({".pdf", ".docx", ".txt"})
PUBLIC_SERVER_ERROR = "An unexpected error occurred."

DEFAULT_CORS_ORIGINS = (
    "http://localhost:3000",
    "http://localhost:3001",
    "https://mangeshraut712.github.io",
    "https://ai-job-helper-steel.vercel.app",
)

_TRUTHY = frozenset({"1", "true", "yes", "on"})


class ResumeUploadError(ValueError):
    """A resume upload the client can correct. `public_message` is safe to return."""

    def __init__(self, message: str, status_code: int = 400) -> None:
        super().__init__(message)
        self.public_message = message
        self.status_code = status_code


def docs_enabled() -> bool:
    """True only for an explicit boolean, not for any non-empty DEBUG string."""
    return os.getenv("DEBUG", "").strip().lower() in _TRUTHY


def resolve_cors_origins(raw: str | None = None) -> list[str]:
    """Return an explicit origin allowlist. `*` and path/credential URLs are rejected."""
    source = os.getenv("CORS_ORIGINS", "") if raw is None else raw
    if not source.strip():
        return list(DEFAULT_CORS_ORIGINS)

    origins: list[str] = []
    for part in source.split(","):
        candidate = part.strip().rstrip("/")
        if not candidate or candidate == "*":
            continue
        try:
            parsed = urlsplit(candidate)
            _ = parsed.port
        except ValueError:
            continue
        if (
            parsed.scheme in {"http", "https"}
            and parsed.hostname
            and parsed.username is None
            and parsed.password is None
            and parsed.path in {"", "/"}
            and not parsed.query
            and not parsed.fragment
        ):
            normalized = f"{parsed.scheme}://{parsed.netloc.lower()}"
            if normalized not in origins:
                origins.append(normalized)
    return origins or list(DEFAULT_CORS_ORIGINS)


def resume_extension(filename: str) -> str:
    if not filename or "." not in filename:
        return ""
    return "." + filename.rsplit(".", 1)[-1].lower()


def validate_resume_filename(filename: str) -> str:
    extension = resume_extension(filename)
    if extension == ".doc":
        raise ResumeUploadError(
            "Legacy .doc files are not supported. Save the resume as .docx, .pdf, or .txt."
        )
    if extension not in ALLOWED_RESUME_EXTENSIONS:
        raise ResumeUploadError(
            "Unsupported file type. Upload a PDF, DOCX, or TXT resume."
        )
    return extension


async def read_limited_upload(
    file: UploadFile,
    max_bytes: int | None = None,
) -> bytes:
    """Read an upload in chunks and reject anything over the size cap."""
    limit = MAX_UPLOAD_BYTES if max_bytes is None else max_bytes
    size = 0
    chunks: list[bytes] = []
    while True:
        chunk = await file.read(64 * 1024)
        if not chunk:
            break
        size += len(chunk)
        if size > limit:
            raise ResumeUploadError("File exceeds the 10MB limit.", status_code=413)
        chunks.append(chunk)
    if size == 0:
        raise ResumeUploadError("Uploaded file is empty.")
    return b"".join(chunks)


def body_size_status(content_length: str | None, limit: int = MAX_REQUEST_BYTES) -> int | None:
    """Return an HTTP status when Content-Length is malformed or over the cap.

    Missing Content-Length is allowed so chunked downloads can still be checked
    by the upload reader. A negative or non-integer length is rejected.
    """
    if content_length is None or not content_length.strip():
        return None
    try:
        size = int(content_length)
    except ValueError:
        return 400
    if size < 0 or size > limit:
        return 413
    return None


def safe_download_name(name: str, extension: str) -> str:
    """Build an ASCII Content-Disposition filename without header or path characters."""
    normalized = unicodedata.normalize("NFKD", name or "")
    cleaned: list[str] = []
    for char in normalized:
        if char.isascii() and (char.isalnum() or char in {" ", "-", "_"}):
            cleaned.append(char)
        elif not unicodedata.combining(char):
            cleaned.append(" ")
    collapsed = "_".join("".join(cleaned).split())
    stem = collapsed[:80] or "resume"
    ext = extension.lower().lstrip(".")
    if ext not in {"pdf", "docx", "txt"}:
        ext = "bin"
    return f"{stem}_resume.{ext}"

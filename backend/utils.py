"""Utility functions for the backend."""
import re
from typing import Optional, List
from urllib.parse import urlparse


def validate_email(email: Optional[str]) -> bool:
    """Validate email address format."""
    if not email or not isinstance(email, str):
        return False
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email.strip()))


def validate_phone(phone: Optional[str]) -> bool:
    """Validate phone number format adhering to standard length constraints (10-15 digits)."""
    if not phone or not isinstance(phone, str):
        return False
    # Remove common separators
    cleaned = re.sub(r'[\s\-\(\)\.]', '', phone)
    # Check if it's all digits (with optional + prefix)
    if cleaned.startswith('+'):
        cleaned = cleaned[1:]
    return cleaned.isdigit() and 10 <= len(cleaned) <= 15


def sanitize_filename(filename: Optional[str]) -> str:
    """Sanitize filename to prevent directory traversal and null-byte injection."""
    if not filename or not isinstance(filename, str):
        return "file"
    # Remove null bytes and control chars
    clean = filename.replace("\x00", "").strip()
    clean = clean.replace("\\", "/").split("/")[-1].strip()
    clean = re.sub(r'[<>:"|?*]', '', clean)
    if clean in {"", ".", ".."}:
        return "file"
    return clean[:255]


def validate_url(url: Optional[str]) -> bool:
    """Validate URL format."""
    if not url or not isinstance(url, str):
        return False
    try:
        result = urlparse(url.strip())
        return all([result.scheme in ['http', 'https'], result.netloc])
    except Exception:
        return False


def truncate_text(text: str, max_length: int = 1000, suffix: str = "...") -> str:
    """Truncate text to maximum length."""
    if max_length < 0:
        raise ValueError(f"max_length must be non-negative, got {max_length}")
    if len(text) <= max_length:
        return text
    if max_length < len(suffix):
        return text[:max_length]
    return text[: max_length - len(suffix)] + suffix


def extract_domain(url: Optional[str]) -> Optional[str]:
    """Extract domain from URL, stripping leading www and dropping ports."""
    if not url or not isinstance(url, str):
        return None
    try:
        parsed = urlparse(url.strip())
        host = (parsed.hostname or parsed.netloc or "").lower()
        if host.startswith("www."):
            host = host[4:]
        return host or None
    except Exception:
        return None


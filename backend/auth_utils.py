from datetime import datetime, timedelta, timezone
import hashlib
import hmac
import os
import secrets
from typing import Optional

import jwt

try:
    import bcrypt
except ImportError:  # pragma: no cover - fallback when bcrypt wheel is absent
    bcrypt = None  # type: ignore[assignment]

try:
    from passlib.context import CryptContext

    pwd_context: Optional[CryptContext] = CryptContext(schemes=["bcrypt"], deprecated="auto")
except Exception:  # pragma: no cover - optional legacy verifier
    pwd_context = None

ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
MIN_SECRET_LENGTH = 32


def get_signing_key() -> str:
    """Return the JWT signing key. Refuse missing or short values."""
    key = os.getenv("SECRET_KEY", "")
    if len(key) < MIN_SECRET_LENGTH:
        raise RuntimeError("SECRET_KEY must be set to at least 32 characters")
    return key


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against a bcrypt (or offline scrypt) hash, rejecting NUL bytes."""
    if not plain_password or not hashed_password or "\x00" in plain_password:
        return False
    pw_bytes = plain_password.encode("utf-8")[:72]
    if hashed_password.startswith("$scrypt$"):
        parts = hashed_password.split("$")
        if len(parts) != 4:
            return False
        try:
            salt = bytes.fromhex(parts[2])
            expected = bytes.fromhex(parts[3])
        except ValueError:
            return False
        derived = hashlib.scrypt(pw_bytes, salt=salt, n=16384, r=8, p=1, dklen=32)
        return hmac.compare_digest(derived, expected)
    if bcrypt is not None:
        try:
            return bool(bcrypt.checkpw(pw_bytes, hashed_password.encode("utf-8")))
        except Exception:
            pass
    if pwd_context is not None:
        try:
            return bool(pwd_context.verify(plain_password, hashed_password))
        except Exception:
            return False
    return False


def get_password_hash(password: str) -> str:
    """Hash a non-empty NUL-free password using bcrypt with 72-byte truncation."""
    if not password or "\x00" in password:
        raise ValueError("Password must be non-empty and cannot contain NUL bytes")
    pw_bytes = password.encode("utf-8")[:72]
    if bcrypt is not None:
        salt = bcrypt.gensalt()
        return bcrypt.hashpw(pw_bytes, salt).decode("utf-8")
    salt_bytes = secrets.token_bytes(16)
    digest = hashlib.scrypt(pw_bytes, salt=salt_bytes, n=16384, r=8, p=1, dklen=32)
    return f"$scrypt${salt_bytes.hex()}${digest.hex()}"


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    lifetime = expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    expire = datetime.now(timezone.utc) + lifetime
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, get_signing_key(), algorithm=ALGORITHM)
    return encoded_jwt

from datetime import datetime, timedelta, timezone
import os
from typing import Optional

import bcrypt
import jwt
from passlib.context import CryptContext

ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
MIN_SECRET_LENGTH = 32

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def get_signing_key() -> str:
    """Return the JWT signing key. Refuse missing or short values."""
    key = os.getenv("SECRET_KEY", "")
    if len(key) < MIN_SECRET_LENGTH:
        raise RuntimeError("SECRET_KEY must be set to at least 32 characters")
    return key


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against a bcrypt hash, falling back to pwd_context."""
    if not plain_password or not hashed_password:
        return False
    try:
        pw_bytes = plain_password.encode("utf-8")[:72]
        return bcrypt.checkpw(pw_bytes, hashed_password.encode("utf-8"))
    except Exception:
        try:
            return pwd_context.verify(plain_password, hashed_password)
        except Exception:
            return False


def get_password_hash(password: str) -> str:
    """Hash password using bcrypt with 72-byte truncation."""
    pw_bytes = (password or "").encode("utf-8")[:72]
    salt = bcrypt.gensalt()
    return bcrypt.hashpw(pw_bytes, salt).decode("utf-8")


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    lifetime = expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    expire = datetime.now(timezone.utc) + lifetime
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, get_signing_key(), algorithm=ALGORITHM)
    return encoded_jwt

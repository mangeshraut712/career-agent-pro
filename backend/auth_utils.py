from datetime import datetime, timedelta, timezone
import os
from typing import Optional

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


def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password):
    return pwd_context.hash(password)


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    lifetime = expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    expire = datetime.now(timezone.utc) + lifetime
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, get_signing_key(), algorithm=ALGORITHM)
    return encoded_jwt

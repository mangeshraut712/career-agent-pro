from datetime import timedelta
import jwt
import pytest
from auth_utils import (
    create_access_token,
    get_password_hash,
    get_signing_key,
    verify_password,
    ALGORITHM,
    MIN_SECRET_LENGTH,
)


def test_get_signing_key_requires_minimum_length(monkeypatch):
    # Missing / empty SECRET_KEY
    monkeypatch.setenv("SECRET_KEY", "")
    with pytest.raises(RuntimeError) as exc_empty:
        get_signing_key()
    assert "at least 32 characters" in str(exc_empty.value)

    # Short SECRET_KEY (31 chars)
    monkeypatch.setenv("SECRET_KEY", "x" * (MIN_SECRET_LENGTH - 1))
    with pytest.raises(RuntimeError) as exc_short:
        get_signing_key()
    assert "at least 32 characters" in str(exc_short.value)

    # Valid length (32 chars)
    valid_key = "a" * MIN_SECRET_LENGTH
    monkeypatch.setenv("SECRET_KEY", valid_key)
    assert get_signing_key() == valid_key


def test_password_hashing_and_verification():
    plain = "SuperSecret123!"
    hashed = get_password_hash(plain)

    assert hashed != plain
    assert verify_password(plain, hashed) is True
    assert verify_password("WrongPassword123!", hashed) is False
    assert verify_password("", hashed) is False
    assert verify_password("SuperSecret123!\x00extra", hashed) is False


def test_password_hashing_rejects_empty_and_nul_bytes():
    with pytest.raises(ValueError, match="non-empty"):
        get_password_hash("")
    with pytest.raises(ValueError, match="NUL"):
        get_password_hash("valid-prefix\x00injected")


def test_create_access_token_default_lifetime(monkeypatch):
    secret = "k" * 32
    monkeypatch.setenv("SECRET_KEY", secret)

    payload = {"sub": "user_123", "role": "admin"}
    token = create_access_token(payload)

    decoded = jwt.decode(token, secret, algorithms=[ALGORITHM])
    assert decoded["sub"] == "user_123"
    assert decoded["role"] == "admin"
    assert "exp" in decoded


def test_create_access_token_custom_lifetime(monkeypatch):
    secret = "k" * 32
    monkeypatch.setenv("SECRET_KEY", secret)

    payload = {"sub": "temp_user"}
    custom_delta = timedelta(minutes=5)
    token = create_access_token(payload, expires_delta=custom_delta)

    decoded = jwt.decode(token, secret, algorithms=[ALGORITHM])
    assert decoded["sub"] == "temp_user"
    assert "exp" in decoded

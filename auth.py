from __future__ import annotations

import base64
import hashlib
import hmac
import json
import secrets
import time
from typing import Any


def _b64encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("utf-8")


def _b64decode(data: str) -> bytes:
    padding = "=" * (-len(data) % 4)
    return base64.urlsafe_b64decode(data + padding)


def hash_password(password: str) -> str:
    """Hash a password with a per-user salt using PBKDF2-HMAC-SHA256."""
    salt = secrets.token_bytes(16)
    hashed = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 120_000)
    return base64.b64encode(salt + hashed).decode("utf-8")


def verify_password(password: str, stored_hash: str) -> bool:
    """Verify a plaintext password against a stored PBKDF2-HMAC-SHA256 hash."""
    raw = base64.b64decode(stored_hash.encode("utf-8"))
    salt = raw[:16]
    expected = raw[16:]
    candidate = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 120_000)
    return hmac.compare_digest(expected, candidate)


def create_token(payload: dict[str, Any], secret: str, ttl_seconds: int) -> str:
    """Create a signed, expiring token payload using HMAC-SHA256."""
    data = dict(payload)
    data["exp"] = int(time.time()) + ttl_seconds
    body = _b64encode(json.dumps(data, separators=(",", ":")).encode("utf-8"))
    signature = hmac.new(secret.encode("utf-8"), body.encode("utf-8"), hashlib.sha256).digest()
    return f"{body}.{_b64encode(signature)}"


def verify_token(token: str, secret: str) -> dict[str, Any]:
    """Verify token signature and expiry, returning the payload if valid."""
    try:
        body, signature = token.split(".", 1)
    except ValueError as exc:  # pragma: no cover
        raise ValueError("Invalid token format.") from exc

    expected = hmac.new(secret.encode("utf-8"), body.encode("utf-8"), hashlib.sha256).digest()
    if not hmac.compare_digest(_b64encode(expected), signature):
        raise ValueError("Invalid token signature.")

    payload = json.loads(_b64decode(body).decode("utf-8"))
    exp = payload.get("exp")
    if exp is None or int(exp) < int(time.time()):
        raise ValueError("Token expired.")

    return payload


def generate_refresh_token() -> str:
    """Generate a high-entropy refresh token for rotation-based sessions."""
    return secrets.token_urlsafe(48)


def hash_token(token: str) -> str:
    """Hash a token using SHA256 for storage and comparison."""
    return hashlib.sha256(token.encode("utf-8")).hexdigest()

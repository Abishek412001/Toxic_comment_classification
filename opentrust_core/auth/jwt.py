"""
JSON Web Token (JWT) Encoding and Decoding for OpenTrust Auth.
"""

import json
import base64
import hmac
import hashlib
import time
from typing import Dict, Any, Optional
from opentrust_core.config import settings
from opentrust_core.exceptions import AuthenticationError


def _base64url_encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("utf-8")


def _base64url_decode(data: str) -> bytes:
    padding = "=" * (4 - (len(data) % 4)) if len(data) % 4 != 0 else ""
    return base64.urlsafe_b64decode((data + padding).encode("utf-8"))


def create_access_token(
    payload: Dict[str, Any],
    expires_in_seconds: Optional[int] = None,
    secret_key: Optional[str] = None,
) -> str:
    """Generates signed JWT Access Token (HS256)."""
    key = (secret_key or settings.SECRET_KEY).encode("utf-8")
    header = {"alg": "HS256", "typ": "JWT"}

    now = int(time.time())
    ttl = expires_in_seconds if expires_in_seconds is not None else settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60

    jwt_payload = {
        **payload,
        "iat": now,
        "exp": now + ttl,
        "iss": settings.PROJECT_NAME,
    }

    header_b64 = _base64url_encode(json.dumps(header, separators=(",", ":")).encode("utf-8"))
    payload_b64 = _base64url_encode(json.dumps(jwt_payload, separators=(",", ":")).encode("utf-8"))

    signing_input = f"{header_b64}.{payload_b64}".encode("utf-8")
    signature = hmac.new(key, signing_input, hashlib.sha256).digest()
    signature_b64 = _base64url_encode(signature)

    return f"{header_b64}.{payload_b64}.{signature_b64}"


def decode_access_token(token: str, secret_key: Optional[str] = None) -> Dict[str, Any]:
    """Decodes and validates signed JWT Access Token (HS256)."""
    key = (secret_key or settings.SECRET_KEY).encode("utf-8")
    parts = token.split(".")
    if len(parts) != 3:
        raise AuthenticationError("Invalid JWT token format")

    header_b64, payload_b64, signature_b64 = parts

    signing_input = f"{header_b64}.{payload_b64}".encode("utf-8")
    expected_sig = hmac.new(key, signing_input, hashlib.sha256).digest()
    actual_sig = _base64url_decode(signature_b64)

    if not hmac.compare_digest(expected_sig, actual_sig):
        raise AuthenticationError("Invalid JWT signature")

    payload_bytes = _base64url_decode(payload_b64)
    payload = json.loads(payload_bytes.decode("utf-8"))

    now = int(time.time())
    if "exp" in payload and payload["exp"] < now:
        raise AuthenticationError("JWT token has expired")

    return payload

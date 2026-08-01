"""
Security, Password Hashing, and JWT Helpers for OpenTrust AI.
"""

import os
import hashlib
import hmac
from typing import Optional


def hash_secret(secret: str, salt: Optional[str] = None) -> str:
    """Generates SHA256 HMAC hash of a secret key or API token."""
    salt_str = salt or "opentrust-salt-2026"
    return hmac.new(salt_str.encode("utf-8"), secret.encode("utf-8"), hashlib.sha256).hexdigest()


def generate_api_key(prefix: str = "ot_live_") -> str:
    """Generates cryptographically strong random API key."""
    random_bytes = os.urandom(24).hex()
    return f"{prefix}{random_bytes}"

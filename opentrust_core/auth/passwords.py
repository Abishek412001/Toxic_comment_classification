"""
Password Hashing and Verification Routines using PBKDF2 HMAC SHA256.
"""

import os
import base64
import hashlib


def hash_password(password: str, iterations: int = 100000) -> str:
    """Hashes plain text password using PBKDF2 HMAC SHA256 with 100,000 iterations and random 16-byte salt."""
    salt = os.urandom(16)
    key = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
    salt_b64 = base64.b64encode(salt).decode("utf-8")
    key_b64 = base64.b64encode(key).decode("utf-8")
    return f"pbkdf2_sha256${iterations}${salt_b64}${key_b64}"


def verify_password(password: str, hashed_password: str) -> bool:
    """Verifies plain text password against PBKDF2 HMAC SHA256 hash."""
    try:
        parts = hashed_password.split("$")
        if len(parts) != 4 or parts[0] != "pbkdf2_sha256":
            return False
        iterations = int(parts[1])
        salt = base64.b64decode(parts[2].encode("utf-8"))
        stored_key = base64.b64decode(parts[3].encode("utf-8"))
        computed_key = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
        return hashlib.sha256(computed_key).digest() == hashlib.sha256(stored_key).digest()
    except Exception:
        return False

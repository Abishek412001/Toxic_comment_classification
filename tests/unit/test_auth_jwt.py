"""
Unit Tests for Password Hashing & JWT Token Encoding/Decoding.
"""

import unittest
from opentrust_core.auth.passwords import hash_password, verify_password
from opentrust_core.auth.jwt import create_access_token, decode_access_token
from opentrust_core.exceptions import AuthenticationError


class TestAuthJWT(unittest.TestCase):
    def test_password_hashing(self):
        pwd = "EnterprisePassword2026!"
        pwd_hash = hash_password(pwd)
        self.assertTrue(verify_password(pwd, pwd_hash))
        self.assertFalse(verify_password("WrongPassword!", pwd_hash))

    def test_jwt_create_and_decode(self):
        payload = {"sub": "usr_test123", "role": "admin", "org_id": "org_789"}
        token = create_access_token(payload, expires_in_seconds=600)
        decoded = decode_access_token(token)

        self.assertEqual(decoded["sub"], "usr_test123")
        self.assertEqual(decoded["role"], "admin")
        self.assertEqual(decoded["org_id"], "org_789")

    def test_jwt_expired_token(self):
        payload = {"sub": "usr_test123"}
        token = create_access_token(payload, expires_in_seconds=-10)
        with self.assertRaises(AuthenticationError):
            decode_access_token(token)


if __name__ == "__main__":
    unittest.main()

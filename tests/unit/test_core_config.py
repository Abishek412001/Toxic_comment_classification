"""
Unit Tests for OpenTrust Core Settings, Exceptions, and Schemas.
"""

import unittest
from opentrust_core.config import settings
from opentrust_core.exceptions import OpenTrustException, ValidationError, AuthenticationError
from opentrust_core.schemas.response import APIResponse, ErrorResponse
from opentrust_core.security import generate_api_key, hash_secret


class TestCoreConfig(unittest.TestCase):
    def test_settings_load(self):
        self.assertEqual(settings.PROJECT_NAME, "OpenTrust AI")
        self.assertIsNotNone(settings.get_database_url())
        self.assertIsNotNone(settings.get_redis_url())

    def test_exceptions_hierarchy(self):
        err = ValidationError("Field is required", details={"field": "username"})
        self.assertEqual(err.status_code, 422)
        self.assertEqual(err.code, "VALIDATION_ERROR")
        self.assertIn("username", err.details.get("field"))

        auth_err = AuthenticationError()
        self.assertEqual(auth_err.status_code, 401)

    def test_security_helpers(self):
        key = generate_api_key()
        self.assertTrue(key.startswith("ot_live_"))
        hashed = hash_secret("secret_token")
        self.assertEqual(len(hashed), 64)

    def test_api_response_schema(self):
        res = APIResponse[str](data="test_data")
        self.assertTrue(res.success)
        self.assertEqual(res.data, "test_data")


if __name__ == "__main__":
    unittest.main()

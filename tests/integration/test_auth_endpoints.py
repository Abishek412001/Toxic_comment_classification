"""
Integration Tests for Auth, Login, Signup, Organizations, and API Keys API Endpoints.
"""

import unittest
from fastapi.testclient import TestClient
from services.api_gateway.app import app


class TestAuthEndpoints(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_signup_and_login_flow(self):
        # 1. Signup
        signup_res = self.client.post(
            "/api/v1/auth/signup",
            json={
                "email": "developer@opentrust.ai",
                "full_name": "Senior Developer",
                "password": "DevPassword2026!",
                "role": "developer",
            },
        )
        self.assertEqual(signup_res.status_code, 201)
        data = signup_res.json()
        self.assertTrue(data["success"])
        self.assertEqual(data["data"]["email"], "developer@opentrust.ai")

        # 2. Login
        login_res = self.client.post(
            "/api/v1/auth/login",
            json={
                "email": "developer@opentrust.ai",
                "password": "DevPassword2026!",
            },
        )
        self.assertEqual(login_res.status_code, 200)
        login_data = login_res.json()
        token = login_data["data"]["access_token"]
        self.assertIsNotNone(token)

        # 3. Get /me
        me_res = self.client.get(
            "/api/v1/auth/me",
            headers={"Authorization": f"Bearer {token}"},
        )
        self.assertEqual(me_res.status_code, 200)
        self.assertEqual(me_res.json()["data"]["email"], "developer@opentrust.ai")

    def test_unauthenticated_me(self):
        res = self.client.get("/api/v1/auth/me")
        self.assertEqual(res.status_code, 401)


if __name__ == "__main__":
    unittest.main()

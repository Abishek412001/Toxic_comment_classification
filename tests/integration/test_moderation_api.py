"""
Integration Tests for Moderation API Gateway Endpoints.
"""

import unittest
from fastapi.testclient import TestClient
from services.api_gateway.app import app


class TestModerationAPI(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_predict_endpoint(self):
        res = self.client.post(
            "/api/v1/moderation/predict",
            json={"text": "Welcome to OpenTrust AI enterprise platform!"},
        )
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertTrue(data["success"])
        self.assertEqual(data["data"]["action"], "PASS")

    def test_batch_predict_endpoint(self):
        res = self.client.post(
            "/api/v1/moderation/batch",
            json={"texts": ["Text 1 benign", "Text 2 toxic kill you stupid"]},
        )
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertTrue(data["success"])
        self.assertEqual(data["data"]["total_processed"], 2)

    def test_policies_endpoint(self):
        res = self.client.get("/api/v1/moderation/policies")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("toxic", data["data"])
        self.assertIn("severe_toxic", data["data"])


if __name__ == "__main__":
    unittest.main()

"""
Integration Tests for XAI API Gateway Endpoints.
"""

import unittest
from fastapi.testclient import TestClient
from services.api_gateway.app import app


class TestXAIAPI(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_explain_endpoint(self):
        res = self.client.post(
            "/api/v1/xai/explain",
            json={"text": "OpenTrust AI XAI explanation endpoint", "method": "shap"},
        )
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertTrue(data["success"])
        self.assertEqual(data["data"]["explainer_method"], "shap")

    def test_batch_explain_endpoint(self):
        res = self.client.post(
            "/api/v1/xai/batch",
            json={"texts": ["Text 1", "Text 2"], "method": "lime"},
        )
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertTrue(data["success"])
        self.assertEqual(data["data"]["total_processed"], 2)


if __name__ == "__main__":
    unittest.main()

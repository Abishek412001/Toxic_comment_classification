"""
Integration Tests for Sentiment API Endpoints.
"""

import unittest
from fastapi.testclient import TestClient
from services.api_gateway.app import app


class TestSentimentAPI(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_sentiment_analyze_endpoint(self):
        res = self.client.post(
            "/api/v1/sentiment/analyze",
            json={"text": "OpenTrust AI platform is awesome!", "engine": "ensemble"},
        )
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertTrue(data["success"])
        self.assertEqual(data["data"]["label"], "POSITIVE")

    def test_sentiment_batch_endpoint(self):
        res = self.client.post(
            "/api/v1/sentiment/batch",
            json={"texts": ["Great product", "Terrible service"], "engine": "vader"},
        )
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertTrue(data["success"])
        self.assertEqual(data["data"]["total_processed"], 2)

    def test_engines_list_endpoint(self):
        res = self.client.get("/api/v1/sentiment/engines")
        self.assertEqual(res.status_code, 200)
        self.assertIn("vader", res.json()["data"])


if __name__ == "__main__":
    unittest.main()

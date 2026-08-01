"""
Integration Tests for Emotion API Endpoints.
"""

import unittest
from fastapi.testclient import TestClient
from services.api_gateway.app import app


class TestEmotionAPI(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_emotion_detect_endpoint(self):
        res = self.client.post(
            "/api/v1/emotion/detect",
            json={"text": "I am so happy and excited!", "top_n": 3},
        )
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertTrue(data["success"])
        self.assertEqual(data["data"]["dominant_emotion"], "joy")

    def test_emotion_batch_endpoint(self):
        res = self.client.post(
            "/api/v1/emotion/batch",
            json={"texts": ["I love this product", "I hate this bug"], "top_n": 2},
        )
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertTrue(data["success"])
        self.assertEqual(data["data"]["total_processed"], 2)

    def test_lexicon_endpoint(self):
        res = self.client.get("/api/v1/emotion/lexicon")
        self.assertEqual(res.status_code, 200)
        self.assertIn("emotions", res.json()["data"])


if __name__ == "__main__":
    unittest.main()

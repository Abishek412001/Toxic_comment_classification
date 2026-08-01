"""
Integration Tests for Decision Intelligence API Gateway Endpoints.
"""

import unittest
from fastapi.testclient import TestClient
from services.api_gateway.app import app


class TestDecisionAPI(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_evaluate_decision_endpoint(self):
        res = self.client.post(
            "/api/v1/decision/evaluate",
            json={
                "text": "Evaluating automated decision API",
                "risk_score": 0.85,
                "confidence_score": 0.92,
                "moderation_action": "FLAG",
            },
        )
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertTrue(data["success"])
        self.assertTrue(data["data"]["trigger_hitl_review"])

    def test_audit_trail_endpoint(self):
        # First post a decision evaluation
        self.client.post(
            "/api/v1/decision/evaluate",
            json={
                "text": "Audit trail log test text",
                "risk_score": 0.20,
                "confidence_score": 0.95,
                "moderation_action": "PASS",
            },
        )
        res = self.client.get("/api/v1/decision/audit-trail")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertTrue(data["success"])
        self.assertGreater(len(data["data"]), 0)


if __name__ == "__main__":
    unittest.main()

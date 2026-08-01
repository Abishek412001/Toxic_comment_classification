"""
Integration Tests for MLOps API Gateway Endpoints.
"""

import unittest
from fastapi.testclient import TestClient
from services.api_gateway.app import app


class TestMLOpsAPI(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)
        # Login as Admin to obtain valid JWT token
        login_res = self.client.post(
            "/api/v1/auth/login",
            json={"email": "admin@opentrust.ai", "password": "AdminSecure2026!"},
        )
        self.token = login_res.json()["data"]["access_token"]
        self.headers = {"Authorization": f"Bearer {self.token}"}

    def test_register_and_promote_model_endpoint(self):
        reg_res = self.client.post(
            "/api/v1/mlops/models/register",
            json={
                "model_name": "sentiment_transformer",
                "version": "1.0.0",
                "framework": "Transformers",
                "metrics": {"accuracy": 0.95},
            },
            headers=self.headers,
        )
        self.assertEqual(reg_res.status_code, 200)
        self.assertTrue(reg_res.json()["success"])

        promo_res = self.client.post(
            "/api/v1/mlops/models/promote",
            json={
                "model_name": "sentiment_transformer",
                "version": "1.0.0",
                "target_stage": "Production",
            },
            headers=self.headers,
        )
        self.assertEqual(promo_res.status_code, 200)
        self.assertTrue(promo_res.json()["success"])

    def test_detect_drift_endpoint(self):
        res = self.client.post(
            "/api/v1/mlops/drift/detect",
            json={"model_name": "toxicity_classifier"},
        )
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertTrue(data["success"])
        self.assertIn("psi_score", data["data"])

    def test_observability_metrics_endpoint(self):
        res = self.client.get("/api/v1/mlops/observability/metrics")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertTrue(data["success"])
        self.assertIn("gpu_utilization_percent", data["data"])


if __name__ == "__main__":
    unittest.main()

"""
Integration Tests for OpenTrust API Gateway Health & Index Endpoints.
"""

import unittest
from fastapi.testclient import TestClient
from services.api_gateway.app import app


class TestGatewayHealth(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_root_endpoint(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        json_data = response.json()
        self.assertEqual(json_data["platform"], "OpenTrust AI")
        self.assertEqual(json_data["status"], "online")

    def test_liveness_probe(self):
        response = self.client.get("/health/liveness")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "UP")

    def test_readiness_probe(self):
        response = self.client.get("/health/readiness")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("memory", data["components"])
        self.assertIn("cpu", data["components"])

    def test_v1_index(self):
        response = self.client.get("/api/v1/")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "online")


if __name__ == "__main__":
    unittest.main()

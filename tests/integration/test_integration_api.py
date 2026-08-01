"""
Integration Tests for Enterprise Integrations & Marketplace API Gateway Endpoints.
"""

import unittest
from fastapi.testclient import TestClient
from services.api_gateway.app import app


class TestIntegrationAPI(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)
        # Login as Admin
        login_res = self.client.post(
            "/api/v1/auth/login",
            json={"email": "admin@opentrust.ai", "password": "AdminSecure2026!"},
        )
        self.token = login_res.json()["data"]["access_token"]
        self.headers = {"Authorization": f"Bearer {self.token}"}

    def test_list_and_dispatch_connectors_endpoint(self):
        list_res = self.client.get("/api/v1/integrations/connectors")
        self.assertEqual(list_res.status_code, 200)
        self.assertTrue(list_res.json()["success"])

        dispatch_res = self.client.post(
            "/api/v1/integrations/connectors/dispatch",
            json={
                "connector_type": "splunk_siem",
                "event_type": "AUDIT_LOG_ALERT",
                "message_payload": {"action": "BLOCK", "reason": "Jailbreak detected"},
            },
            headers=self.headers,
        )
        self.assertEqual(dispatch_res.status_code, 200)
        self.assertTrue(dispatch_res.json()["success"])
        self.assertEqual(dispatch_res.json()["data"]["connector_type"], "splunk_siem")

    def test_marketplace_list_and_install_endpoint(self):
        items_res = self.client.get("/api/v1/integrations/marketplace/items")
        self.assertEqual(items_res.status_code, 200)
        self.assertTrue(items_res.json()["success"])

        install_res = self.client.post(
            "/api/v1/integrations/marketplace/install",
            json={"item_id": "mp_eu_gdpr"},
            headers=self.headers,
        )
        self.assertEqual(install_res.status_code, 200)
        self.assertTrue(install_res.json()["success"])


if __name__ == "__main__":
    unittest.main()

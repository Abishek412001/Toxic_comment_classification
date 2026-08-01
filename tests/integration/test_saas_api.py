"""
Integration Tests for SaaS API Gateway Endpoints.
"""

import unittest
from fastapi.testclient import TestClient
from services.api_gateway.app import app


class TestSaaSAPI(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)
        # Login as Admin
        login_res = self.client.post(
            "/api/v1/auth/login",
            json={"email": "admin@opentrust.ai", "password": "AdminSecure2026!"},
        )
        self.token = login_res.json()["data"]["access_token"]
        self.headers = {"Authorization": f"Bearer {self.token}"}

    def test_subscribe_and_current_flow(self):
        sub_res = self.client.post(
            "/api/v1/saas/subscriptions/subscribe",
            json={
                "organization_id": "org_opentrust_root",
                "plan": "Enterprise",
                "billing_cycle": "annual",
                "allocated_seats": 25,
            },
            headers=self.headers,
        )
        self.assertEqual(sub_res.status_code, 200)
        self.assertTrue(sub_res.json()["success"])
        self.assertEqual(sub_res.json()["data"]["plan"], "Enterprise")

        current_res = self.client.get(
            "/api/v1/saas/subscriptions/current",
            headers=self.headers,
        )
        self.assertEqual(current_res.status_code, 200)
        self.assertEqual(current_res.json()["data"]["plan"], "Enterprise")

    def test_billing_usage_and_invoices_endpoint(self):
        usage_res = self.client.get(
            "/api/v1/saas/billing/usage",
            headers=self.headers,
        )
        self.assertEqual(usage_res.status_code, 200)
        self.assertTrue(usage_res.json()["success"])

        inv_res = self.client.get(
            "/api/v1/saas/billing/invoices",
            headers=self.headers,
        )
        self.assertEqual(inv_res.status_code, 200)
        self.assertTrue(inv_res.json()["success"])
        self.assertTrue(inv_res.json()["data"]["invoice_id"].startswith("INV-2026-"))


if __name__ == "__main__":
    unittest.main()

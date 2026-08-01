"""
Unit Tests for SaaS Subscriptions, Metering, Billing Invoices & Webhooks.
"""

import unittest
from services.saas_service.engine import SaaSEngine
from services.saas_service.schemas import (
    SubscriptionRequest,
    SubscriptionPlanEnum,
    BillingCycleEnum,
    WebhookNotificationRequest,
)


class TestSaaSService(unittest.TestCase):
    def setUp(self):
        self.engine = SaaSEngine()

    def test_subscription_creation(self):
        req = SubscriptionRequest(
            organization_id="org_test_123",
            plan=SubscriptionPlanEnum.BUSINESS,
            billing_cycle=BillingCycleEnum.ANNUAL,
            allocated_seats=10,
        )
        res = self.engine.create_subscription(req)
        self.assertEqual(res.plan, SubscriptionPlanEnum.BUSINESS)
        self.assertEqual(res.allocated_seats, 10)
        self.assertEqual(res.status, "ACTIVE")

    def test_usage_metrics(self):
        res = self.engine.get_usage_metrics("org_test_123")
        self.assertGreater(res.api_calls_count, 0)
        self.assertGreater(res.tokens_processed_count, 0)

    def test_invoice_generation(self):
        res = self.engine.generate_invoice("org_test_123")
        self.assertTrue(res.invoice_id.startswith("INV-2026-"))
        self.assertGreater(res.total_usd, 0.0)
        self.assertGreater(len(res.line_items), 0)

    def test_webhook_dispatch(self):
        req = WebhookNotificationRequest(
            target_url="https://api.customer.com/webhooks/opentrust",
            event_type="USAGE_ALERT_80",
            payload={"org_id": "org_test_123", "usage_percent": 82.5},
        )
        res = self.engine.dispatch_webhook(req)
        self.assertTrue(res.delivered)
        self.assertEqual(res.status_code, 200)


if __name__ == "__main__":
    unittest.main()

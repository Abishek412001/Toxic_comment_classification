"""
Unit Tests for Connectors & Marketplace Engine.
"""

import unittest
from services.integration_service.engine import IntegrationEngine
from services.integration_service.schemas import (
    ConnectorDispatchRequest,
    ConnectorTypeEnum,
    MarketplaceInstallRequest,
)


class TestIntegrationService(unittest.TestCase):
    def setUp(self):
        self.engine = IntegrationEngine()

    def test_list_connectors(self):
        connectors = self.engine.list_connectors()
        self.assertGreater(len(connectors), 0)
        types = [c["connector_type"] for c in connectors]
        self.assertIn("slack", types)
        self.assertIn("splunk_siem", types)

    def test_dispatch_connector_event(self):
        req = ConnectorDispatchRequest(
            connector_type=ConnectorTypeEnum.SLACK,
            event_type="FLAGGED_TOXIC_CONTENT",
            message_payload={"text": "Severe toxic comment detected", "user": "user123"},
        )
        res = self.engine.dispatch_connector_event(req)
        self.assertEqual(res.status, "DELIVERED")
        self.assertEqual(res.connector_type, ConnectorTypeEnum.SLACK)

    def test_marketplace_items_and_installation(self):
        items = self.engine.list_marketplace_items()
        self.assertGreater(len(items), 0)

        install_req = MarketplaceInstallRequest(item_id="mp_hipaa_health")
        install_res = self.engine.install_marketplace_item(install_req)
        self.assertEqual(install_res.status, "INSTALLED")
        self.assertEqual(install_res.item_id, "mp_hipaa_health")


if __name__ == "__main__":
    unittest.main()

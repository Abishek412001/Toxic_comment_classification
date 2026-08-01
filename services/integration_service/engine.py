"""
Enterprise Integration & Marketplace Orchestration Engine.
"""

from typing import List, Dict, Any
from services.integration_service.connectors import connector_manager
from services.integration_service.marketplace import marketplace_manager
from services.integration_service.schemas import (
    ConnectorDispatchRequest,
    ConnectorDispatchResponse,
    MarketplaceItemResponse,
    MarketplaceInstallRequest,
    MarketplaceInstallResponse,
)


class IntegrationEngine:
    """Enterprise Integrations & Marketplace Orchestrator."""

    def dispatch_connector_event(self, request: ConnectorDispatchRequest) -> ConnectorDispatchResponse:
        return connector_manager.dispatch_event(request)

    def list_connectors(self) -> List[Dict[str, Any]]:
        return connector_manager.list_connectors()

    def list_marketplace_items(self) -> List[MarketplaceItemResponse]:
        return marketplace_manager.list_items()

    def install_marketplace_item(self, request: MarketplaceInstallRequest) -> MarketplaceInstallResponse:
        return marketplace_manager.install_item(request)


integration_engine = IntegrationEngine()

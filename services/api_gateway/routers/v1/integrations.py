"""
Enterprise Integrations, Connectors & Marketplace API Gateway Router.
"""

from typing import List, Dict, Any
from fastapi import APIRouter, Depends
from services.integration_service.schemas import (
    ConnectorDispatchRequest,
    ConnectorDispatchResponse,
    MarketplaceItemResponse,
    MarketplaceInstallRequest,
    MarketplaceInstallResponse,
)
from services.integration_service.engine import integration_engine
from opentrust_core.auth.dependencies import get_current_user, require_role
from opentrust_core.auth.models import UserRead, RoleEnum
from opentrust_core.schemas.response import APIResponse

router = APIRouter(prefix="/integrations", tags=["Enterprise Integrations & Marketplace"])


@router.get("/connectors", response_model=APIResponse[List[Dict[str, Any]]])
async def list_connectors():
    """Lists all available enterprise connectors (Slack, Teams, Zendesk, Splunk, GitHub)."""
    res = integration_engine.list_connectors()
    return APIResponse[List[Dict[str, Any]]](data=res)


@router.post("/connectors/dispatch", response_model=APIResponse[ConnectorDispatchResponse])
async def dispatch_connector_event(
    request: ConnectorDispatchRequest,
    current_user: UserRead = Depends(require_role([RoleEnum.ADMIN, RoleEnum.DEVELOPER])),
):
    """Dispatches safety alert or audit log to enterprise integration connector."""
    res = integration_engine.dispatch_connector_event(request)
    return APIResponse[ConnectorDispatchResponse](
        data=res,
        message=f"Event dispatched to {request.connector_type.value}.",
    )


@router.get("/marketplace/items", response_model=APIResponse[List[MarketplaceItemResponse]])
async def list_marketplace_items():
    """Lists available policy packs and prompt safety templates in the Marketplace."""
    res = integration_engine.list_marketplace_items()
    return APIResponse[List[MarketplaceItemResponse]](data=res)


@router.post("/marketplace/install", response_model=APIResponse[MarketplaceInstallResponse])
async def install_marketplace_item(
    request: MarketplaceInstallRequest,
    current_user: UserRead = Depends(require_role([RoleEnum.ADMIN])),
):
    """Installs a policy pack or prompt safety template from Marketplace."""
    res = integration_engine.install_marketplace_item(request)
    return APIResponse[MarketplaceInstallResponse](
        data=res,
        message=f"Installed '{res.item_name}' (v{res.version}).",
    )

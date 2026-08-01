"""
SaaS Subscriptions, Metering, Billing & Webhooks API Gateway Router.
"""

from typing import List
from fastapi import APIRouter, Depends
from services.saas_service.schemas import (
    SubscriptionRequest,
    SubscriptionResponse,
    UsageMetricsResponse,
    InvoiceResponse,
    WebhookNotificationRequest,
    WebhookNotificationResponse,
)
from services.saas_service.engine import saas_engine
from opentrust_core.auth.dependencies import get_current_user, require_role
from opentrust_core.auth.models import UserRead, RoleEnum
from opentrust_core.schemas.response import APIResponse

router = APIRouter(prefix="/saas", tags=["SaaS Subscriptions, Metering & Billing"])


@router.post("/subscriptions/subscribe", response_model=APIResponse[SubscriptionResponse])
async def subscribe_tenant(
    request: SubscriptionRequest,
    current_user: UserRead = Depends(require_role([RoleEnum.ADMIN])),
):
    """Subscribes an organization to a SaaS plan tier."""
    res = saas_engine.create_subscription(request)
    return APIResponse[SubscriptionResponse](
        data=res,
        message=f"Organization subscribed to {request.plan.value} plan.",
    )


@router.get("/subscriptions/current", response_model=APIResponse[SubscriptionResponse])
async def get_current_subscription(current_user: UserRead = Depends(get_current_user)):
    """Retrieves active subscription and plan details for organization."""
    res = saas_engine.get_subscription(current_user.organization_id or "org_default")
    return APIResponse[SubscriptionResponse](data=res)


@router.get("/billing/usage", response_model=APIResponse[UsageMetricsResponse])
async def get_billing_usage(current_user: UserRead = Depends(get_current_user)):
    """Retrieves metered usage analytics for tenant organization."""
    res = saas_engine.get_usage_metrics(current_user.organization_id or "org_default")
    return APIResponse[UsageMetricsResponse](data=res)


@router.get("/billing/invoices", response_model=APIResponse[InvoiceResponse])
async def get_billing_invoice(current_user: UserRead = Depends(get_current_user)):
    """Generates tax-ready billing invoice for current usage period."""
    res = saas_engine.generate_invoice(current_user.organization_id or "org_default")
    return APIResponse[InvoiceResponse](data=res)


@router.post("/notifications/webhook", response_model=APIResponse[WebhookNotificationResponse])
async def dispatch_webhook(
    request: WebhookNotificationRequest,
    current_user: UserRead = Depends(require_role([RoleEnum.ADMIN, RoleEnum.DEVELOPER])),
):
    """Dispatches webhook notification payload to target URL."""
    res = saas_engine.dispatch_webhook(request)
    return APIResponse[WebhookNotificationResponse](
        data=res,
        message="Webhook notification dispatched successfully.",
    )

"""
SaaS Subscription, Plan Quota, and License Manager.
"""

import uuid
from datetime import datetime, timedelta
from typing import Dict, Any
from services.saas_service.schemas import (
    SubscriptionRequest,
    SubscriptionResponse,
    SubscriptionPlanEnum,
    BillingCycleEnum,
)
from opentrust_core.exceptions import NotFoundError

PLAN_PRICING: Dict[SubscriptionPlanEnum, Dict[str, Any]] = {
    SubscriptionPlanEnum.STARTER: {"price_monthly": 49.0, "api_limit": 50000},
    SubscriptionPlanEnum.PROFESSIONAL: {"price_monthly": 299.0, "api_limit": 500000},
    SubscriptionPlanEnum.BUSINESS: {"price_monthly": 899.0, "api_limit": 2000000},
    SubscriptionPlanEnum.ENTERPRISE: {"price_monthly": 2499.0, "api_limit": 10000000},
}

SUBSCRIPTION_STORE: Dict[str, Dict[str, Any]] = {}


class SubscriptionManager:
    """Enterprise SaaS Subscription & Plan Quota Manager."""

    def create_subscription(self, request: SubscriptionRequest) -> SubscriptionResponse:
        """Subscribes an organization to a SaaS plan tier."""
        sub_id = f"sub_{uuid.uuid4().hex[:8]}"
        plan_info = PLAN_PRICING.get(request.plan, {"price_monthly": 299.0, "api_limit": 500000})

        discount = 0.85 if request.billing_cycle == BillingCycleEnum.ANNUAL else 1.0
        final_price = round(plan_info["price_monthly"] * discount, 2)
        expires_at = datetime.utcnow() + timedelta(days=365 if request.billing_cycle == BillingCycleEnum.ANNUAL else 30)

        sub_record = {
            "subscription_id": sub_id,
            "organization_id": request.organization_id,
            "plan": request.plan,
            "billing_cycle": request.billing_cycle,
            "status": "ACTIVE",
            "monthly_price_usd": final_price,
            "allocated_seats": request.allocated_seats,
            "monthly_api_limit": plan_info["api_limit"],
            "started_at": datetime.utcnow(),
            "expires_at": expires_at,
        }

        SUBSCRIPTION_STORE[request.organization_id] = sub_record

        return SubscriptionResponse(**sub_record)

    def get_subscription(self, organization_id: str) -> SubscriptionResponse:
        """Fetches active subscription for an organization."""
        sub = SUBSCRIPTION_STORE.get(organization_id)
        if not sub:
            # Default Starter Subscription
            return self.create_subscription(
                SubscriptionRequest(organization_id=organization_id, plan=SubscriptionPlanEnum.STARTER)
            )
        return SubscriptionResponse(**sub)


subscription_manager = SubscriptionManager()

"""
Enterprise SaaS Engine Combining Subscriptions, Metered Billing & Webhooks.
"""

from typing import List
from services.saas_service.subscription import subscription_manager
from services.saas_service.billing import billing_manager
from services.saas_service.notifications import notification_dispatcher
from services.saas_service.schemas import (
    SubscriptionRequest,
    SubscriptionResponse,
    UsageMetricsResponse,
    InvoiceResponse,
    WebhookNotificationRequest,
    WebhookNotificationResponse,
)


class SaaSEngine:
    """Enterprise SaaS Commercialization & Metering Orchestrator."""

    def create_subscription(self, request: SubscriptionRequest) -> SubscriptionResponse:
        return subscription_manager.create_subscription(request)

    def get_subscription(self, organization_id: str) -> SubscriptionResponse:
        return subscription_manager.get_subscription(organization_id)

    def get_usage_metrics(self, organization_id: str) -> UsageMetricsResponse:
        return billing_manager.get_usage_metrics(organization_id)

    def generate_invoice(self, organization_id: str) -> InvoiceResponse:
        return billing_manager.generate_invoice(organization_id)

    def dispatch_webhook(self, request: WebhookNotificationRequest) -> WebhookNotificationResponse:
        return notification_dispatcher.dispatch_webhook(request)


saas_engine = SaaSEngine()

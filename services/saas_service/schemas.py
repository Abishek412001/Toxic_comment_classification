"""
Pydantic v2 Schemas for SaaS Subscriptions, Billing Invoices, and Usage Metering.
"""

from enum import Enum
from typing import List, Dict, Optional, Any
from datetime import datetime
from pydantic import Field
from opentrust_core.schemas.base import BaseSchema


class SubscriptionPlanEnum(str, Enum):
    STARTER = "Starter"
    PROFESSIONAL = "Professional"
    BUSINESS = "Business"
    ENTERPRISE = "Enterprise"


class BillingCycleEnum(str, Enum):
    MONTHLY = "monthly"
    ANNUAL = "annual"


class SubscriptionRequest(BaseSchema):
    organization_id: str
    plan: SubscriptionPlanEnum = SubscriptionPlanEnum.PROFESSIONAL
    billing_cycle: BillingCycleEnum = BillingCycleEnum.MONTHLY
    allocated_seats: int = Field(default=5, ge=1, le=1000)


class SubscriptionResponse(BaseSchema):
    subscription_id: str
    organization_id: str
    plan: SubscriptionPlanEnum
    billing_cycle: BillingCycleEnum
    status: str  # ACTIVE, TRIAL, CANCELLED
    monthly_price_usd: float
    allocated_seats: int
    monthly_api_limit: int
    started_at: datetime = Field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None


class UsageMetricsResponse(BaseSchema):
    organization_id: str
    period_start: datetime
    period_end: datetime
    api_calls_count: int
    tokens_processed_count: int
    documents_moderated_count: int
    storage_used_mb: float
    gpu_hours_used: float
    quota_consumption_percent: float


class InvoiceLineItem(BaseSchema):
    description: str
    amount_usd: float


class InvoiceResponse(BaseSchema):
    invoice_id: str
    organization_id: str
    billing_period: str
    subtotal_usd: float
    tax_usd: float
    total_usd: float
    status: str  # PAID, PENDING, OVERDUE
    line_items: List[InvoiceLineItem]
    issued_at: datetime = Field(default_factory=datetime.utcnow)
    due_date: datetime


class WebhookNotificationRequest(BaseSchema):
    target_url: str = Field(pattern=r"^https?://")
    event_type: str  # USAGE_ALERT_80, USAGE_ALERT_100, INVOICE_PAID, SECURITY_INCIDENT
    payload: Dict[str, Any]


class WebhookNotificationResponse(BaseSchema):
    dispatch_id: str
    target_url: str
    status_code: int = 200
    delivered: bool = True
    timestamp: datetime = Field(default_factory=datetime.utcnow)

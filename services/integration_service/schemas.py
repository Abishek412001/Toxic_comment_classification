"""
Pydantic v2 Schemas for Enterprise Connectors and Marketplace Policies.
"""

from enum import Enum
from typing import List, Dict, Optional, Any
from datetime import datetime
from pydantic import Field
from opentrust_core.schemas.base import BaseSchema


class ConnectorTypeEnum(str, Enum):
    SLACK = "slack"
    TEAMS = "microsoft_teams"
    ZENDESK = "zendesk"
    SPLUNK = "splunk_siem"
    GITHUB = "github_actions"


class ConnectorDispatchRequest(BaseSchema):
    connector_type: ConnectorTypeEnum
    event_type: str = "FLAGGED_TOXIC_CONTENT"
    message_payload: Dict[str, Any]


class ConnectorDispatchResponse(BaseSchema):
    dispatch_id: str
    connector_type: ConnectorTypeEnum
    status: str  # DELIVERED, QUEUED, FAILED
    latency_ms: float
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class MarketplaceItemResponse(BaseSchema):
    item_id: str
    name: str
    category: str  # POLICY_PACK, PROMPT_SAFETY, COMPLIANCE_RULE
    version: str
    author: str
    rating: float
    downloads_count: int
    installed: bool = False


class MarketplaceInstallRequest(BaseSchema):
    item_id: str
    version: Optional[str] = None


class MarketplaceInstallResponse(BaseSchema):
    install_id: str
    item_id: str
    item_name: str
    version: str
    status: str  # INSTALLED, ALREADY_INSTALLED
    installed_at: datetime = Field(default_factory=datetime.utcnow)

"""
Health Check Schema Models.
"""

from typing import Dict, Any, Optional
from pydantic import Field
from opentrust_core.schemas.base import BaseSchema


class HealthComponent(BaseSchema):
    """Component-level health check detail."""

    status: str = "UP"  # UP, DOWN, DEGRADED
    details: Optional[Dict[str, Any]] = None


class HealthStatus(BaseSchema):
    """System-wide health probe status."""

    status: str = "UP"  # UP, DOWN, DEGRADED
    service: str = "OpenTrust AI API Gateway"
    version: str = "1.0.0"
    environment: str = "development"
    components: Dict[str, HealthComponent] = Field(default_factory=dict)

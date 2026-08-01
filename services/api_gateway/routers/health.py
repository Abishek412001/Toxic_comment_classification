"""
Health Probe Endpoints for API Gateway.
"""

from fastapi import APIRouter
from opentrust_core.health import HealthChecker
from opentrust_core.schemas.health import HealthStatus

router = APIRouter(prefix="/health", tags=["Health & Diagnostics"])


@router.get("/liveness", response_model=HealthStatus, summary="Liveness Probe")
async def liveness_probe():
    """Returns basic process liveness status."""
    return HealthChecker.get_liveness()


@router.get("/readiness", response_model=HealthStatus, summary="Readiness Probe")
async def readiness_probe():
    """Returns deep readiness status checking system memory, CPU, and disk space."""
    return HealthChecker.get_readiness()

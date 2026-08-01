"""
Moderation Service Endpoints: Real-time Prediction, Batch Processing, and Policy Inspection.
"""

from fastapi import APIRouter, Depends
from services.moderation_service.schemas import (
    ModerationRequest,
    ModerationResponse,
    BatchModerationRequest,
    BatchModerationResponse,
)
from services.moderation_service.engine import moderation_engine
from services.moderation_service.policy import DEFAULT_CATEGORY_THRESHOLDS
from opentrust_core.auth.dependencies import check_api_key_rate_limit
from opentrust_core.schemas.response import APIResponse

router = APIRouter(prefix="/moderation", tags=["Content Moderation Engine"])


@router.post("/predict", response_model=APIResponse[ModerationResponse])
async def predict_moderation(
    request: ModerationRequest,
    rate_info: dict = Depends(check_api_key_rate_limit),
):
    """Evaluates multi-label toxicity scores and returns moderation action for a single text."""
    res = moderation_engine.predict_single(request)
    return APIResponse[ModerationResponse](
        data=res,
        message="Moderation analysis completed successfully.",
    )


@router.post("/batch", response_model=APIResponse[BatchModerationResponse])
async def predict_moderation_batch(
    request: BatchModerationRequest,
    rate_info: dict = Depends(check_api_key_rate_limit),
):
    """Evaluates bulk texts in parallel and returns aggregated moderation actions."""
    res = moderation_engine.predict_batch(request)
    return APIResponse[BatchModerationResponse](
        data=res,
        message=f"Batch moderation completed for {res.total_processed} items.",
    )


@router.get("/policies", response_model=APIResponse[dict])
async def get_moderation_policies():
    """Returns active tenant policy threshold configuration."""
    formatted_policies = {
        cat: {"flag_threshold": flag, "block_threshold": block}
        for cat, (flag, block) in DEFAULT_CATEGORY_THRESHOLDS.items()
    }
    return APIResponse[dict](data=formatted_policies)

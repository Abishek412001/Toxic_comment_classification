"""
XAI Service API Gateway Router.
"""

from fastapi import APIRouter, Depends
from services.xai_service.schemas import (
    XAIRequest,
    XAIResponse,
    BatchXAIRequest,
    BatchXAIResponse,
)
from services.xai_service.engine import xai_engine
from opentrust_core.auth.dependencies import check_api_key_rate_limit
from opentrust_core.schemas.response import APIResponse

router = APIRouter(prefix="/xai", tags=["Explainable AI (XAI)"])


@router.post("/explain", response_model=APIResponse[XAIResponse])
async def explain_prediction(
    request: XAIRequest,
    rate_info: dict = Depends(check_api_key_rate_limit),
):
    """Generates SHAP or LIME feature contribution attributions and HTML visualization export."""
    res = xai_engine.explain_single(request)
    return APIResponse[XAIResponse](
        data=res,
        message="XAI feature explanation generated successfully.",
    )


@router.post("/batch", response_model=APIResponse[BatchXAIResponse])
async def explain_prediction_batch(
    request: BatchXAIRequest,
    rate_info: dict = Depends(check_api_key_rate_limit),
):
    """Generates batch feature explanations across bulk texts."""
    res = xai_engine.explain_batch(request)
    return APIResponse[BatchXAIResponse](
        data=res,
        message=f"Batch XAI explanations completed for {res.total_processed} items.",
    )

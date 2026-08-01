"""
MLOps, Model Registry, Drift & Observability API Gateway Router.
"""

from fastapi import APIRouter, Depends
from services.mlops_service.schemas import (
    ModelRegistrationRequest,
    ModelRegistrationResponse,
    ModelPromotionRequest,
    ModelRollbackRequest,
    DriftDetectionRequest,
    DriftDetectionResponse,
    RetrainingTriggerRequest,
    RetrainingTriggerResponse,
    ObservabilityMetricsResponse,
)
from services.mlops_service.engine import mlops_engine
from opentrust_core.auth.dependencies import check_api_key_rate_limit, require_role
from opentrust_core.auth.models import RoleEnum
from opentrust_core.schemas.response import APIResponse

router = APIRouter(prefix="/mlops", tags=["Enterprise MLOps & Observability"])


@router.post("/models/register", response_model=APIResponse[ModelRegistrationResponse])
async def register_model(
    request: ModelRegistrationRequest,
    current_user: dict = Depends(require_role([RoleEnum.ADMIN, RoleEnum.DEVELOPER])),
):
    """Registers a new model version into the OpenTrust Model Registry."""
    res = mlops_engine.register_model(request)
    return APIResponse[ModelRegistrationResponse](
        data=res,
        message="Model version registered successfully.",
    )


@router.post("/models/promote", response_model=APIResponse[ModelRegistrationResponse])
async def promote_model(
    request: ModelPromotionRequest,
    current_user: dict = Depends(require_role([RoleEnum.ADMIN])),
):
    """Promotes a model version to Production or Staging."""
    res = mlops_engine.promote_model(request)
    return APIResponse[ModelRegistrationResponse](
        data=res,
        message=f"Model version {request.version} promoted to {request.target_stage.value}.",
    )


@router.post("/models/rollback", response_model=APIResponse[ModelRegistrationResponse])
async def rollback_model(
    request: ModelRollbackRequest,
    current_user: dict = Depends(require_role([RoleEnum.ADMIN])),
):
    """Instantly rolls back Production to previous model version."""
    res = mlops_engine.rollback_model(request)
    return APIResponse[ModelRegistrationResponse](
        data=res,
        message=f"Production model rolled back successfully to version {res.version}.",
    )


@router.post("/drift/detect", response_model=APIResponse[DriftDetectionResponse])
async def detect_drift(
    request: DriftDetectionRequest,
    rate_info: dict = Depends(check_api_key_rate_limit),
):
    """Calculates population stability index (PSI) and data/concept drift metrics."""
    res = mlops_engine.detect_drift(request)
    return APIResponse[DriftDetectionResponse](
        data=res,
        message="Drift analysis completed.",
    )


@router.post("/retrain/trigger", response_model=APIResponse[RetrainingTriggerResponse])
async def trigger_retraining(
    request: RetrainingTriggerRequest,
    current_user: dict = Depends(require_role([RoleEnum.ADMIN, RoleEnum.DEVELOPER])),
):
    """Triggers automated model retraining pipeline."""
    res = mlops_engine.trigger_retraining(request)
    return APIResponse[RetrainingTriggerResponse](
        data=res,
        message="Automated retraining pipeline triggered.",
    )


@router.get("/observability/metrics", response_model=APIResponse[ObservabilityMetricsResponse])
async def get_observability_metrics():
    """Retrieves real-time platform observability and GPU/CPU cost metrics."""
    res = mlops_engine.get_observability_metrics()
    return APIResponse[ObservabilityMetricsResponse](data=res)

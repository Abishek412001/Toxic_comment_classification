"""
Decision Intelligence & Audit Trail API Gateway Router.
"""

from typing import List
from fastapi import APIRouter, Depends
from services.decision_service.schemas import (
    DecisionEvaluationRequest,
    DecisionEvaluationResponse,
    AuditLogEntry,
)
from services.decision_service.engine import decision_engine
from opentrust_core.auth.dependencies import check_api_key_rate_limit
from opentrust_core.schemas.response import APIResponse

router = APIRouter(prefix="/decision", tags=["Decision Intelligence & Audit"])


@router.post("/evaluate", response_model=APIResponse[DecisionEvaluationResponse])
async def evaluate_decision(
    request: DecisionEvaluationRequest,
    rate_info: dict = Depends(check_api_key_rate_limit),
):
    """Evaluates prediction confidence, triggers HITL review if needed, and logs immutable audit entry."""
    res = decision_engine.evaluate_decision(request)
    return APIResponse[DecisionEvaluationResponse](
        data=res,
        message="Decision evaluation completed and logged to audit trail.",
    )


@router.get("/audit-trail", response_model=APIResponse[List[AuditLogEntry]])
async def get_audit_trail(limit: int = 50):
    """Returns recent decision audit trail entries."""
    entries = decision_engine.get_audit_trail(limit=limit)
    return APIResponse[List[AuditLogEntry]](data=entries)

"""
Pydantic v2 Schemas for Decision Intelligence, Audit Trail, and HITL Triggers.
"""

from typing import List, Dict, Optional, Any
from datetime import datetime
from pydantic import Field
from opentrust_core.schemas.base import BaseSchema


class DecisionEvaluationRequest(BaseSchema):
    text: str = Field(min_length=1, max_length=10000)
    risk_score: float = Field(ge=0.0, le=1.0)
    confidence_score: float = Field(ge=0.0, le=1.0)
    moderation_action: str = Field(description="PASS, FLAG, or BLOCK")


class DecisionEvaluationResponse(BaseSchema):
    decision_id: str
    final_action: str
    trigger_hitl_review: bool
    hitl_reason: Optional[str] = None
    risk_level: str  # LOW, MEDIUM, HIGH, CRITICAL
    recommendation: str
    audit_logged: bool
    evaluated_at: datetime = Field(default_factory=datetime.utcnow)


class AuditLogEntry(BaseSchema):
    decision_id: str
    text_snippet: str
    final_action: str
    risk_score: float
    confidence_score: float
    trigger_hitl_review: bool
    timestamp: datetime = Field(default_factory=datetime.utcnow)

"""
Pydantic v2 Schemas for Moderation Requests, Responses, and Policy Signals.
"""

from enum import Enum
from typing import List, Dict, Optional, Any
from pydantic import Field
from opentrust_core.schemas.base import BaseSchema


class ActionEnum(str, Enum):
    PASS = "PASS"
    FLAG = "FLAG"
    BLOCK = "BLOCK"


class ToxicityScores(BaseSchema):
    toxic: float = Field(ge=0.0, le=1.0)
    severe_toxic: float = Field(ge=0.0, le=1.0)
    obscene: float = Field(ge=0.0, le=1.0)
    threat: float = Field(ge=0.0, le=1.0)
    insult: float = Field(ge=0.0, le=1.0)
    identity_hate: float = Field(ge=0.0, le=1.0)


class ModerationRequest(BaseSchema):
    text: str = Field(min_length=1, max_length=10000, description="Text string to evaluate")
    model_id: Optional[str] = Field(default="distilbert", description="Model architecture identifier")
    metadata: Optional[Dict[str, Any]] = None


class ModerationResponse(BaseSchema):
    text: str
    flagged: bool
    action: ActionEnum
    overall_risk_score: float = Field(ge=0.0, le=1.0)
    categories: ToxicityScores
    flagged_categories: List[str]
    latency_ms: float


class BatchModerationRequest(BaseSchema):
    texts: List[str] = Field(min_items=1, max_items=500, description="List of text strings for bulk evaluation")
    model_id: Optional[str] = Field(default="distilbert", description="Model architecture identifier")


class BatchModerationResponse(BaseSchema):
    total_processed: int
    flagged_count: int
    blocked_count: int
    results: List[ModerationResponse]
    batch_latency_ms: float

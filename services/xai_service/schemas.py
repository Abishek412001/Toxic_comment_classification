"""
Pydantic v2 Schemas for XAI Feature Attributions and Explanations.
"""

from enum import Enum
from typing import List, Optional, Dict, Any
from pydantic import Field
from opentrust_core.schemas.base import BaseSchema


class ExplainerMethodEnum(str, Enum):
    SHAP = "shap"
    LIME = "lime"
    INTEGRATED_GRADIENTS = "integrated_gradients"


class FeatureContribution(BaseSchema):
    feature: str
    score: float
    importance: float


class XAIRequest(BaseSchema):
    text: str = Field(min_length=1, max_length=10000, description="Text input to explain")
    method: ExplainerMethodEnum = Field(default=ExplainerMethodEnum.SHAP)
    top_features: int = Field(default=5, ge=1, le=20)


class XAIResponse(BaseSchema):
    text: str
    explainer_method: ExplainerMethodEnum
    prediction: str
    prediction_confidence: float
    feature_contributions: List[FeatureContribution]
    explanation_summary: str
    html_export: Optional[str] = None
    latency_ms: float


class BatchXAIRequest(BaseSchema):
    texts: List[str] = Field(min_items=1, max_items=50)
    method: ExplainerMethodEnum = Field(default=ExplainerMethodEnum.SHAP)
    top_features: int = Field(default=5, ge=1, le=20)


class BatchXAIResponse(BaseSchema):
    total_processed: int
    results: List[XAIResponse]
    batch_latency_ms: float

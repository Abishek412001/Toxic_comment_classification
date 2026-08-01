"""
Pydantic v2 Schemas for Emotion Detection Requests, Responses, and Distributions.
"""

from typing import List, Dict, Optional, Any
from pydantic import Field
from opentrust_core.schemas.base import BaseSchema


class EmotionDistribution(BaseSchema):
    anger: float = Field(ge=0.0, le=1.0)
    anticipation: float = Field(ge=0.0, le=1.0)
    disgust: float = Field(ge=0.0, le=1.0)
    fear: float = Field(ge=0.0, le=1.0)
    joy: float = Field(ge=0.0, le=1.0)
    sadness: float = Field(ge=0.0, le=1.0)
    surprise: float = Field(ge=0.0, le=1.0)
    trust: float = Field(ge=0.0, le=1.0)


class EmotionItem(BaseSchema):
    emotion: str
    probability: float = Field(ge=0.0, le=1.0)


class EmotionRequest(BaseSchema):
    text: str = Field(min_length=1, max_length=10000, description="Text string to evaluate for emotions")
    top_n: int = Field(default=3, ge=1, le=8, description="Number of top emotion probabilities to return")


class EmotionResponse(BaseSchema):
    text: str
    dominant_emotion: str
    confidence: float = Field(ge=0.0, le=1.0)
    top_emotions: List[EmotionItem]
    distribution: EmotionDistribution
    latency_ms: float


class BatchEmotionRequest(BaseSchema):
    texts: List[str] = Field(min_items=1, max_items=500, description="List of text strings for bulk emotion detection")
    top_n: int = Field(default=3, ge=1, le=8)


class BatchEmotionResponse(BaseSchema):
    total_processed: int
    emotion_counts: Dict[str, int]
    results: List[EmotionResponse]
    batch_latency_ms: float

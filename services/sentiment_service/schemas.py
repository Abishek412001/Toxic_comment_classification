"""
Pydantic v2 Schemas for Sentiment Analysis Requests, Responses, and Scores.
"""

from enum import Enum
from typing import List, Optional, Any, Dict
from pydantic import Field
from opentrust_core.schemas.base import BaseSchema


class SentimentLabelEnum(str, Enum):
    POSITIVE = "POSITIVE"
    NEUTRAL = "NEUTRAL"
    NEGATIVE = "NEGATIVE"


class EngineTypeEnum(str, Enum):
    VADER = "vader"
    TEXTBLOB = "textblob"
    TRANSFORMER = "transformer"
    ENSEMBLE = "ensemble"


class SentimentRequest(BaseSchema):
    text: str = Field(min_length=1, max_length=10000, description="Text string to analyze")
    engine: EngineTypeEnum = Field(default=EngineTypeEnum.ENSEMBLE, description="Sentiment analysis engine to use")


class SentimentResponse(BaseSchema):
    text: str
    label: SentimentLabelEnum
    compound_score: float = Field(ge=-1.0, le=1.0, description="Compound sentiment score [-1.0 to 1.0]")
    polarity: float = Field(ge=-1.0, le=1.0, description="Polarity score [-1.0 to 1.0]")
    subjectivity: float = Field(ge=0.0, le=1.0, description="Subjectivity score [0.0 to 1.0]")
    confidence: float = Field(ge=0.0, le=1.0, description="Model prediction confidence score")
    engine_used: EngineTypeEnum
    latency_ms: float


class BatchSentimentRequest(BaseSchema):
    texts: List[str] = Field(min_items=1, max_items=500, description="List of text strings for bulk sentiment analysis")
    engine: EngineTypeEnum = Field(default=EngineTypeEnum.ENSEMBLE)


class BatchSentimentResponse(BaseSchema):
    total_processed: int
    positive_count: int
    neutral_count: int
    negative_count: int
    results: List[SentimentResponse]
    batch_latency_ms: float

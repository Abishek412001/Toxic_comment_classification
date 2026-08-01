"""
OpenTrust AI - Enterprise Emotion Intelligence Microservice Package.
"""

from services.emotion_service.engine import EmotionEngine
from services.emotion_service.schemas import (
    EmotionRequest,
    EmotionResponse,
    BatchEmotionRequest,
    BatchEmotionResponse,
    EmotionDistribution,
    EmotionItem,
)

__all__ = [
    "EmotionEngine",
    "EmotionRequest",
    "EmotionResponse",
    "BatchEmotionRequest",
    "BatchEmotionResponse",
    "EmotionDistribution",
    "EmotionItem",
]

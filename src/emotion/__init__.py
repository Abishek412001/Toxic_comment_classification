"""
Enterprise Emotion Mining Package (Phase 8).

Provides production-grade NRC Lexicon and Transformer emotion analyzers, pipeline, factory, and evaluators.
"""

from src.emotion.exceptions import (
    EmotionError,
    ValidationError,
    EmotionAnalysisError,
    ConfigurationError,
)
from src.emotion.config import EmotionConfig
from src.emotion.base_emotion import BaseEmotionAnalyzer
from src.emotion.emotion_factory import EmotionFactory
from src.emotion.emotion_pipeline import EmotionPipeline

__all__ = [
    "EmotionError",
    "ValidationError",
    "EmotionAnalysisError",
    "ConfigurationError",
    "EmotionConfig",
    "BaseEmotionAnalyzer",
    "EmotionFactory",
    "EmotionPipeline",
]

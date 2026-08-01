"""
Feature Engineering Package.

Provides production-grade, modular, configurable feature extraction architecture,
base classes, validators, factory, utilities, and pipeline execution modules.
"""

from src.features.exceptions import (
    FeatureEngineeringError,
    FeatureExtractionError,
    ConfigurationError,
    EmbeddingModelError,
    ValidationError,
)
from src.features.config import FeatureConfig, ConfigurationManager
from src.features.base_feature_extractor import BaseFeatureExtractor
from src.features.feature_validator import FeatureValidator
from src.features.feature_factory import FeatureFactory
from src.features.feature_pipeline import FeaturePipeline

__all__ = [
    "FeatureEngineeringError",
    "FeatureExtractionError",
    "ConfigurationError",
    "EmbeddingModelError",
    "ValidationError",
    "FeatureConfig",
    "ConfigurationManager",
    "BaseFeatureExtractor",
    "FeatureValidator",
    "FeatureFactory",
    "FeaturePipeline",
]

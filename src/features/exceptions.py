"""
Custom Exceptions Module for Feature Engineering.

Defines exception hierarchy for validation, configuration, extraction, and embedding model errors.
"""


class FeatureEngineeringError(Exception):
    """Base exception class for all feature engineering errors."""
    pass


class FeatureExtractionError(FeatureEngineeringError):
    """Raised when an error occurs during vector generation or feature transformation."""
    pass


class ConfigurationError(FeatureEngineeringError):
    """Raised when invalid or conflicting feature configuration options are supplied."""
    pass


class EmbeddingModelError(FeatureEngineeringError):
    """Raised when deep learning or transformer embedding models fail to load or infer."""
    pass


class ValidationError(FeatureEngineeringError):
    """Raised when feature inputs, shapes, or vocabulary checks fail validation."""
    pass

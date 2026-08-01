"""
Custom Exception Hierarchy for Emotion Mining Framework (Phase 8).
"""


class EmotionError(Exception):
    """Base exception class for all emotion framework errors."""
    pass


class ValidationError(EmotionError):
    """Raised when input text sequences fail validation."""
    pass


class EmotionAnalysisError(EmotionError):
    """Raised when emotion analysis engines fail during execution."""
    pass


class ConfigurationError(EmotionError):
    """Raised when invalid emotion configuration options are provided."""
    pass

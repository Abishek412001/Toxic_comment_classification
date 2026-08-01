"""
Custom Exception Hierarchy for Explainable AI Framework (Phase 9).
"""


class XAIError(Exception):
    """Base exception class for all XAI framework errors."""
    pass


class ValidationError(XAIError):
    """Raised when input text, model objects, or prediction arrays fail validation."""
    pass


class ExplanationError(XAIError):
    """Raised when SHAP or LIME explanation engines encounter calculation errors."""
    pass


class ConfigurationError(XAIError):
    """Raised when invalid XAI configuration parameters are provided."""
    pass

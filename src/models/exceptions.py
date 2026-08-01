"""
Custom Exception Hierarchy for Model Development (Phase 5).
"""


class ModelError(Exception):
    """Base exception class for all model development errors."""
    pass


class TrainingError(ModelError):
    """Raised when an error occurs during model fitting or training loops."""
    pass


class PredictionError(ModelError):
    """Raised when an error occurs during batch or real-time inference prediction."""
    pass


class EvaluationError(ModelError):
    """Raised when evaluation metric computation or cross-validation fails."""
    pass


class ConfigurationError(ModelError):
    """Raised when invalid model configuration parameters are supplied."""
    pass

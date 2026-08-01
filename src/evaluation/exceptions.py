"""
Custom Exception Hierarchy for Multi-Label Classification Evaluation (Phase 6).
"""


class EvaluationError(Exception):
    """Base exception class for all evaluation framework errors."""
    pass


class ValidationError(EvaluationError):
    """Raised when input predictions, probability matrices, or target label shapes fail validation."""
    pass


class ThresholdOptimizationError(EvaluationError):
    """Raised when per-label probability threshold optimization fails."""
    pass


class MetricCalculationError(EvaluationError):
    """Raised when evaluation metric calculations encounter division errors or invalid arrays."""
    pass


class ConfigurationError(EvaluationError):
    """Raised when invalid evaluation configuration parameters are supplied."""
    pass

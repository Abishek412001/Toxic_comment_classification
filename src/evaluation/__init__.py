"""
Multi-Label Evaluation Package (Phase 6).

Provides production-grade multi-label evaluators, metric calculators, threshold optimizers,
ROC AUC analyzers, confusion matrix analyzers, error analyzers, and reporting modules.
"""

from src.evaluation.exceptions import (
    EvaluationError,
    ValidationError,
    ThresholdOptimizationError,
    MetricCalculationError,
    ConfigurationError,
)
from src.evaluation.config import EvaluationConfig
from src.evaluation.evaluator import BaseEvaluator
from src.evaluation.metrics import MetricsCalculator
from src.evaluation.threshold_optimizer import ThresholdOptimizer
from src.evaluation.multilabel_pipeline import EvaluationPipeline

__all__ = [
    "EvaluationError",
    "ValidationError",
    "ThresholdOptimizationError",
    "MetricCalculationError",
    "ConfigurationError",
    "EvaluationConfig",
    "BaseEvaluator",
    "MetricsCalculator",
    "ThresholdOptimizer",
    "EvaluationPipeline",
]

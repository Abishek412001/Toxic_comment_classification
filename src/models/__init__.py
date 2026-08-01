"""
Model Development Package (Phase 5).

Provides production-grade multi-label models, data splitting strategies, baselines,
trainers, predictors, evaluators, experiment trackers, and registry modules.
"""

from src.models.exceptions import (
    ModelError,
    TrainingError,
    PredictionError,
    EvaluationError,
    ConfigurationError,
)
from src.models.config import ModelConfig
from src.models.base_model import BaseModel
from src.models.model_factory import ModelFactory
from src.models.trainer import ModelTrainer
from src.models.predictor import ModelPredictor
from src.models.evaluator import ModelEvaluator
from src.models.experiment_tracker import ExperimentTracker
from src.models.model_registry import ModelRegistry

__all__ = [
    "ModelError",
    "TrainingError",
    "PredictionError",
    "EvaluationError",
    "ConfigurationError",
    "ModelConfig",
    "BaseModel",
    "ModelFactory",
    "ModelTrainer",
    "ModelPredictor",
    "ModelEvaluator",
    "ExperimentTracker",
    "ModelRegistry",
]

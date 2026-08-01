"""
Base Evaluator Abstract Interface Module.

Defines evaluate, evaluate_per_label, save, and load contracts.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import numpy as np


class BaseEvaluator(ABC):
    """Abstract Base Class for all multi-label evaluation modules following SOLID principles."""

    def __init__(self, name: str = "BaseEvaluator"):
        """Initializes evaluator with component name.

        Args:
            name: Component name identifier.
        """
        self.name = name

    @abstractmethod
    def evaluate(self, y_true: np.ndarray, y_proba: np.ndarray, threshold: float = 0.5) -> Dict[str, Any]:
        """Calculates global multi-label evaluation metrics.

        Args:
            y_true: True binary label matrix (N x 6).
            y_proba: Predicted probability matrix (N x 6).
            threshold: Probability decision threshold.

        Returns:
            Dict containing metric scores.
        """
        pass

    @abstractmethod
    def evaluate_per_label(self, y_true: np.ndarray, y_proba: np.ndarray, threshold: float = 0.5) -> Dict[str, Dict[str, float]]:
        """Calculates per-label metrics across all target classes.

        Args:
            y_true: True binary label matrix.
            y_proba: Predicted probability matrix.
            threshold: Probability decision threshold.

        Returns:
            Dict mapping each target label to metric dictionary.
        """
        pass

    @abstractmethod
    def save(self, filepath: str) -> None:
        """Serializes evaluation report to disk."""
        pass

    @abstractmethod
    def load(self, filepath: str) -> "BaseEvaluator":
        """Deserializes evaluation report from disk."""
        pass

"""
Base Model Abstract Interface Module.

Defines standard fit, predict, predict_proba, save, and load contracts.
"""

from abc import ABC, abstractmethod
from typing import List, Any, Optional
import numpy as np


class BaseModel(ABC):
    """Abstract Base Class for all multi-label model architectures following SOLID principles."""

    def __init__(self, name: str = "BaseModel"):
        """Initializes model with name identifier.

        Args:
            name: Component identifier name.
        """
        self.name = name
        self.is_fitted = False

    @abstractmethod
    def fit(self, X: Any, y: Any) -> "BaseModel":
        """Fits model on feature matrix X and multi-label targets y.

        Args:
            X: Input feature matrix (Sparse or Dense).
            y: Target label matrix (N x 6).

        Returns:
            Fitted BaseModel instance.
        """
        pass

    @abstractmethod
    def predict(self, X: Any, threshold: float = 0.5) -> np.ndarray:
        """Predicts binary label indicators (0 or 1) for input matrix X.

        Args:
            X: Input feature matrix.
            threshold: Probability decision threshold.

        Returns:
            Binary matrix of shape (N, 6).
        """
        pass

    @abstractmethod
    def predict_proba(self, X: Any) -> np.ndarray:
        """Predicts continuous probabilities (0.0 to 1.0) for input matrix X.

        Args:
            X: Input feature matrix.

        Returns:
            Probability matrix of shape (N, 6).
        """
        pass

    @abstractmethod
    def save(self, filepath: str) -> None:
        """Serializes model artifact to disk.

        Args:
            filepath: Target file path.
        """
        pass

    @abstractmethod
    def load(self, filepath: str) -> "BaseModel":
        """Deserializes model artifact from disk.

        Args:
            filepath: Target file path.

        Returns:
            Loaded BaseModel instance.
        """
        pass

"""
Base Explainer Abstract Interface Module.

Defines explain_instance, explain_batch, save, and load contracts following SOLID principles.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional


class BaseExplainer(ABC):
    """Abstract Base Class for all Explainable AI engines."""

    def __init__(self, name: str = "BaseExplainer"):
        """Initializes explainer with method identifier.

        Args:
            name: Explainer component name.
        """
        self.name = name

    @abstractmethod
    def explain_instance(self, text: str, model: Any, target_label: Optional[str] = None) -> Dict[str, Any]:
        """Explains a single prediction instance.

        Args:
            text: Input text string.
            model: Trained classifier model.
            target_label: Optional label identifier to explain.

        Returns:
            Dict containing prediction, confidence, feature_importance, positive_contributors, and negative_contributors.
        """
        pass

    @abstractmethod
    def explain_batch(self, texts: List[str], model: Any) -> List[Dict[str, Any]]:
        """Explains a batch of prediction instances.

        Args:
            texts: List of text strings.
            model: Trained classifier model.

        Returns:
            List of result dictionaries.
        """
        pass

    @abstractmethod
    def save(self, filepath: str) -> None:
        """Serializes explainer parameters."""
        pass

    @abstractmethod
    def load(self, filepath: str) -> "BaseExplainer":
        """Deserializes explainer parameters."""
        pass

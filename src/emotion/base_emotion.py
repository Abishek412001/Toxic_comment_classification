"""
Base Emotion Analyzer Abstract Interface Module.

Defines analyze, analyze_batch, save, and load contracts following SOLID principles.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional


class BaseEmotionAnalyzer(ABC):
    """Abstract Base Class for all emotion mining engines."""

    def __init__(self, name: str = "BaseEmotionAnalyzer"):
        """Initializes analyzer with engine identifier.

        Args:
            name: Engine component name.
        """
        self.name = name

    @abstractmethod
    def analyze(self, text: str) -> Dict[str, Any]:
        """Analyzes a single text string.

        Args:
            text: Input string.

        Returns:
            Dict containing emotion_label, confidence_score, probabilities, and top_emotions.
        """
        pass

    @abstractmethod
    def analyze_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Analyzes a batch of text strings.

        Args:
            texts: List of text strings.

        Returns:
            List of result dictionaries.
        """
        pass

    @abstractmethod
    def save(self, filepath: str) -> None:
        """Serializes analyzer parameters or model weights."""
        pass

    @abstractmethod
    def load(self, filepath: str) -> "BaseEmotionAnalyzer":
        """Deserializes analyzer parameters or model weights."""
        pass

"""
Base Feature Extractor Abstract Interface Module.

Defines standard fit, transform, fit_transform, save, load, and get_feature_names contracts.
"""

from abc import ABC, abstractmethod
from typing import List, Any, Optional, Tuple
import numpy as np


class BaseFeatureExtractor(ABC):
    """Abstract Base Class for all feature extractors following SOLID principles."""

    def __init__(self, name: str = "BaseFeatureExtractor"):
        """Initializes extractor with component name.

        Args:
            name: Extractor component name.
        """
        self.name = name
        self.is_fitted = False

    @abstractmethod
    def fit(self, texts: List[str]) -> "BaseFeatureExtractor":
        """Fits vectorizer or embedding model on text corpus.

        Args:
            texts: List of text strings.

        Returns:
            Fitted extractor instance.
        """
        pass

    @abstractmethod
    def transform(self, texts: List[str]) -> Any:
        """Transforms text corpus into feature matrix.

        Args:
            texts: List of text strings.

        Returns:
            Sparse matrix or Dense NumPy array.
        """
        pass

    def fit_transform(self, texts: List[str]) -> Any:
        """Fits extractor and transforms text corpus in a single pass.

        Args:
            texts: List of text strings.

        Returns:
            Sparse matrix or Dense NumPy array.
        """
        return self.fit(texts).transform(texts)

    @abstractmethod
    def save(self, filepath: str) -> None:
        """Serializes fitted extractor artifact to file.

        Args:
            filepath: Target file path.
        """
        pass

    @abstractmethod
    def load(self, filepath: str) -> "BaseFeatureExtractor":
        """Deserializes extractor artifact from file.

        Args:
            filepath: Target file path.

        Returns:
            Loaded extractor instance.
        """
        pass

    @abstractmethod
    def get_feature_names(self) -> List[str]:
        """Returns list of feature names or embedding dimension labels.

        Returns:
            List of feature names.
        """
        pass

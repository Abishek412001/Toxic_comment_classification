"""
Base TextCleaner Interface Module.

Abstract base class defining standard transformer methods (transform, transform_batch)
for all preprocessing sub-modules following SOLID Open-Closed & Dependency Inversion principles.
"""

from abc import ABC, abstractmethod
from typing import List, Any
from src.preprocessing.validator import TextValidator


class TextCleaner(ABC):
    """Abstract Base Class for text cleaning transformers."""

    def __init__(self, name: str = "TextCleaner"):
        """Initializes transformer with a descriptive name.

        Args:
            name: Transformer component name.
        """
        self.name = name

    @abstractmethod
    def transform(self, text: str) -> str:
        """Transforms a single text string.

        Args:
            text: Input string.

        Returns:
            Processed output string.
        """
        pass

    def transform_batch(self, texts: List[str]) -> List[str]:
        """Transforms a list of text strings.

        Args:
            texts: List of input strings.

        Returns:
            List of processed strings.
        """
        if not isinstance(texts, (list, tuple)):
            texts = [texts]

        return [self.transform(t) for t in texts]

    def validate_input(self, text: Any, allow_empty: bool = True) -> str:
        """Utility wrapper for input validation.

        Args:
            text: Input text candidate.
            allow_empty: Whether empty text is allowed.

        Returns:
            Validated text string.
        """
        return TextValidator.validate_text(text, allow_empty=allow_empty)

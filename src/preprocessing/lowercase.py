"""
Lowercasing Transformer Module (Step 27).

Provides production-grade LowercaseTransformer, apply_lowercase(),
batch_lowercase(), and validate_input() functions.
"""

import logging
from typing import List, Any
from src.preprocessing.text_cleaner import TextCleaner
from src.preprocessing.exceptions import CleaningError, InvalidInputError

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class LowercaseTransformer(TextCleaner):
    """Transformer for converting text to lowercase while preserving numbers, punctuation, and Unicode accents."""

    def __init__(self):
        super().__init__(name="LowercaseTransformer")

    def transform(self, text: str) -> str:
        """Applies lowercasing to text.

        Args:
            text: Input text string.

        Returns:
            Lowercased string.
        """
        validated_text = self.validate_input(text, allow_empty=True)
        try:
            return validated_text.lower()
        except Exception as e:
            logger.error(f"Error in LowercaseTransformer: {e}")
            raise CleaningError(f"Lowercasing failed: {e}") from e

    def batch_transform(self, texts: List[str]) -> List[str]:
        """Applies lowercasing across a batch of strings.

        Args:
            texts: List of input text strings.

        Returns:
            List of lowercased strings.
        """
        return self.transform_batch(texts)


def apply_lowercase(text: str) -> str:
    """Functional wrapper for single string lowercasing."""
    return LowercaseTransformer().transform(text)


def batch_lowercase(texts: List[str]) -> List[str]:
    """Functional wrapper for batch lowercasing."""
    return LowercaseTransformer().transform_batch(texts)


def validate_input(text: Any) -> str:
    """Wrapper function for text validation."""
    return LowercaseTransformer().validate_input(text, allow_empty=True)

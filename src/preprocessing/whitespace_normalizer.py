"""
Whitespace Normalization Module (Step 36).

Collapses multiple spaces, tabs, newlines, and leading/trailing whitespace into single spaces.
"""

import re
import logging
from typing import List
from src.preprocessing.text_cleaner import TextCleaner
from src.preprocessing.exceptions import CleaningError
from src.preprocessing.constants import WHITESPACE_REGEX

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class WhitespaceNormalizer(TextCleaner):
    """Transformer for normalizing whitespace formatting in text."""

    def __init__(self):
        super().__init__(name="WhitespaceNormalizer")

    def transform(self, text: str) -> str:
        """Normalizes whitespace in text.

        Args:
            text: Input string.

        Returns:
            Clean single-spaced string.
        """
        validated_text = self.validate_input(text, allow_empty=True)
        if not validated_text:
            return ""

        try:
            return WHITESPACE_REGEX.sub(" ", validated_text).strip()
        except Exception as e:
            logger.error(f"Error in WhitespaceNormalizer: {e}")
            raise CleaningError(f"Whitespace normalization failed: {e}") from e

    def batch_normalize(self, texts: List[str]) -> List[str]:
        """Normalizes whitespace across a batch of strings."""
        return self.transform_batch(texts)


def normalize_whitespace(text: str) -> str:
    """Functional wrapper for whitespace normalization."""
    return WhitespaceNormalizer().transform(text)


def batch_normalize_whitespace(texts: List[str]) -> List[str]:
    """Functional wrapper for batch whitespace normalization."""
    return WhitespaceNormalizer().transform_batch(texts)

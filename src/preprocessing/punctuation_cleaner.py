"""
Punctuation Removal Module (Step 34).

Strips standard ASCII and Unicode punctuation marks while preserving Unicode words and spaces.
"""

import re
import string
import logging
from typing import List, Optional
from src.preprocessing.text_cleaner import TextCleaner
from src.preprocessing.exceptions import CleaningError

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class PunctuationCleaner(TextCleaner):
    """Transformer for stripping punctuation characters from text."""

    def __init__(self, preserve_chars: Optional[str] = None):
        super().__init__(name="PunctuationCleaner")
        punct = string.punctuation
        if preserve_chars:
            for c in preserve_chars:
                punct = punct.replace(c, "")
        self.punctuation_pattern = re.compile(f"[{re.escape(punct)}]")

    def transform(self, text: str) -> str:
        """Strips punctuation marks from text.

        Args:
            text: Input string.

        Returns:
            Clean string without punctuation.
        """
        validated_text = self.validate_input(text, allow_empty=True)
        if not validated_text:
            return ""

        try:
            cleaned = self.punctuation_pattern.sub(" ", validated_text)
            return re.sub(r"\s+", " ", cleaned).strip()
        except Exception as e:
            logger.error(f"Error in PunctuationCleaner: {e}")
            raise CleaningError(f"Punctuation removal failed: {e}") from e

    def batch_remove_punctuation(self, texts: List[str]) -> List[str]:
        """Strips punctuation across a batch of strings."""
        return self.transform_batch(texts)


def remove_punctuation(text: str, preserve_chars: Optional[str] = None) -> str:
    """Functional wrapper for punctuation removal."""
    return PunctuationCleaner(preserve_chars=preserve_chars).transform(text)


def batch_remove_punctuation(texts: List[str], preserve_chars: Optional[str] = None) -> List[str]:
    """Functional wrapper for batch punctuation removal."""
    return PunctuationCleaner(preserve_chars=preserve_chars).transform_batch(texts)

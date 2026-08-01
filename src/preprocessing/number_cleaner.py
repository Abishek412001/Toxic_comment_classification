"""
Number Removal & Normalization Module (Step 33).

Strips standalone integers and decimals or normalizes them to zero / replacement tokens.
"""

import re
import logging
from typing import List
from src.preprocessing.text_cleaner import TextCleaner
from src.preprocessing.exceptions import CleaningError
from src.preprocessing.constants import NUMBER_REGEX

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class NumberCleaner(TextCleaner):
    """Transformer for removing or replacing standalone numbers in text."""

    def __init__(self, replacement_token: str = "0", remove_entirely: bool = False):
        super().__init__(name="NumberCleaner")
        self.replacement_token = replacement_token
        self.remove_entirely = remove_entirely

    def transform(self, text: str) -> str:
        """Removes or replaces standalone digits in text.

        Args:
            text: Input string.

        Returns:
            Processed string.
        """
        validated_text = self.validate_input(text, allow_empty=True)
        if not validated_text:
            return ""

        try:
            replacement = "" if self.remove_entirely else f" {self.replacement_token} "
            cleaned = NUMBER_REGEX.sub(replacement, validated_text)
            return re.sub(r"\s+", " ", cleaned).strip()
        except Exception as e:
            logger.error(f"Error in NumberCleaner: {e}")
            raise CleaningError(f"Number removal failed: {e}") from e

    def batch_remove_numbers(self, texts: List[str]) -> List[str]:
        """Removes or replaces numbers across a batch of strings."""
        return self.transform_batch(texts)


def remove_numbers(text: str, replacement_token: str = "0", remove_entirely: bool = False) -> str:
    """Functional wrapper for number cleaning."""
    return NumberCleaner(replacement_token=replacement_token, remove_entirely=remove_entirely).transform(text)


def batch_remove_numbers(texts: List[str], replacement_token: str = "0", remove_entirely: bool = False) -> List[str]:
    """Functional wrapper for batch number cleaning."""
    return NumberCleaner(replacement_token=replacement_token, remove_entirely=remove_entirely).transform_batch(texts)

"""
URL Removal Module (Step 30).

Detects and strips or replaces URLs (HTTP, HTTPS, WWW, FTP) with configurable tokens.
"""

import re
import logging
from typing import List
from src.preprocessing.text_cleaner import TextCleaner
from src.preprocessing.exceptions import CleaningError
from src.preprocessing.constants import URL_REGEX

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class URLCleaner(TextCleaner):
    """Transformer for removing or replacing web URLs in text."""

    def __init__(self, replacement_token: str = "[URL]"):
        super().__init__(name="URLCleaner")
        self.replacement_token = replacement_token

    def transform(self, text: str) -> str:
        """Strips or replaces URLs in text.

        Args:
            text: Input string.

        Returns:
            Processed string.
        """
        validated_text = self.validate_input(text, allow_empty=True)
        if not validated_text:
            return ""

        try:
            replacement = f" {self.replacement_token} " if self.replacement_token else " "
            cleaned = URL_REGEX.sub(replacement, validated_text)
            return re.sub(r"\s+", " ", cleaned).strip()
        except Exception as e:
            logger.error(f"Error in URLCleaner: {e}")
            raise CleaningError(f"URL removal failed: {e}") from e

    def batch_remove_urls(self, texts: List[str]) -> List[str]:
        """Strips or replaces URLs across a batch of strings."""
        return self.transform_batch(texts)


def remove_urls(text: str, replacement_token: str = "[URL]") -> str:
    """Functional wrapper for URL removal."""
    return URLCleaner(replacement_token=replacement_token).transform(text)


def batch_remove_urls(texts: List[str], replacement_token: str = "[URL]") -> List[str]:
    """Functional wrapper for batch URL removal."""
    return URLCleaner(replacement_token=replacement_token).transform_batch(texts)

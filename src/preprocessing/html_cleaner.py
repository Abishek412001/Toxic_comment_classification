"""
HTML Tag Removal Module (Step 29).

Strips HTML/XML tags while preserving text content and handling malformed HTML.
"""

import re
import logging
from typing import List
from src.preprocessing.text_cleaner import TextCleaner
from src.preprocessing.exceptions import CleaningError
from src.preprocessing.constants import HTML_TAG_REGEX

try:
    from bs4 import BeautifulSoup
    HAS_BS4 = True
except ImportError:
    HAS_BS4 = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class HTMLCleaner(TextCleaner):
    """Transformer for removing HTML markup tags and unescaping HTML entities."""

    def __init__(self, use_bs4: bool = True):
        super().__init__(name="HTMLCleaner")
        self.use_bs4 = use_bs4 and HAS_BS4

    def transform(self, text: str) -> str:
        """Strips HTML markup from text.

        Args:
            text: Input string.

        Returns:
            Clean text string without HTML tags.
        """
        validated_text = self.validate_input(text, allow_empty=True)
        if not validated_text or "<" not in validated_text:
            return validated_text

        try:
            if self.use_bs4:
                soup = BeautifulSoup(validated_text, "html.parser")
                clean_text = soup.get_text(separator=" ")
            else:
                clean_text = HTML_TAG_REGEX.sub(" ", validated_text)

            # Normalize residual spaces
            return re.sub(r"\s+", " ", clean_text).strip()
        except Exception as e:
            logger.error(f"Error in HTMLCleaner: {e}")
            raise CleaningError(f"HTML cleaning failed: {e}") from e

    def batch_remove_html(self, texts: List[str]) -> List[str]:
        """Strips HTML tags across a batch of strings."""
        return self.transform_batch(texts)


def remove_html_tags(text: str) -> str:
    """Functional wrapper for HTML tag removal."""
    return HTMLCleaner().transform(text)


def batch_remove_html(texts: List[str]) -> List[str]:
    """Functional wrapper for batch HTML removal."""
    return HTMLCleaner().transform_batch(texts)

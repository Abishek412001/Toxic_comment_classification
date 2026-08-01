"""
Special Character Removal Module (Step 35).

Strips non-alphanumeric special symbols while supporting configurable allowlists.
"""

import re
import logging
from typing import List, Optional
from src.preprocessing.text_cleaner import TextCleaner
from src.preprocessing.exceptions import CleaningError

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class SpecialCharacterCleaner(TextCleaner):
    """Transformer for removing non-linguistic special symbols."""

    def __init__(self, allowlist: Optional[str] = None):
        super().__init__(name="SpecialCharacterCleaner")
        self.allowlist = allowlist or ""
        escaped_allow = re.escape(self.allowlist)
        self.pattern = re.compile(f"[^a-zA-Z0-9\\s{escaped_allow}]")

    def transform(self, text: str) -> str:
        """Strips special characters from text.

        Args:
            text: Input string.

        Returns:
            Clean string.
        """
        validated_text = self.validate_input(text, allow_empty=True)
        if not validated_text:
            return ""

        try:
            cleaned = self.pattern.sub(" ", validated_text)
            return re.sub(r"\s+", " ", cleaned).strip()
        except Exception as e:
            logger.error(f"Error in SpecialCharacterCleaner: {e}")
            raise CleaningError(f"Special character cleaning failed: {e}") from e

    def batch_remove_special(self, texts: List[str]) -> List[str]:
        """Strips special characters across a batch of strings."""
        return self.transform_batch(texts)


def remove_special_characters(text: str, allowlist: Optional[str] = None) -> str:
    """Functional wrapper for special character removal."""
    return SpecialCharacterCleaner(allowlist=allowlist).transform(text)


def batch_remove_special_characters(texts: List[str], allowlist: Optional[str] = None) -> List[str]:
    """Functional wrapper for batch special character removal."""
    return SpecialCharacterCleaner(allowlist=allowlist).transform_batch(texts)

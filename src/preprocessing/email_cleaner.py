"""
Email Address Removal Module (Step 31).

Detects and strips or replaces email addresses with configurable tokens for PII privacy.
"""

import re
import logging
from typing import List
from src.preprocessing.text_cleaner import TextCleaner
from src.preprocessing.exceptions import CleaningError
from src.preprocessing.constants import EMAIL_REGEX

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class EmailCleaner(TextCleaner):
    """Transformer for removing or replacing email addresses in text."""

    def __init__(self, replacement_token: str = "[EMAIL]"):
        super().__init__(name="EmailCleaner")
        self.replacement_token = replacement_token

    def transform(self, text: str) -> str:
        """Strips or replaces email addresses in text.

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
            cleaned = EMAIL_REGEX.sub(replacement, validated_text)
            return re.sub(r"\s+", " ", cleaned).strip()
        except Exception as e:
            logger.error(f"Error in EmailCleaner: {e}")
            raise CleaningError(f"Email removal failed: {e}") from e

    def batch_remove_emails(self, texts: List[str]) -> List[str]:
        """Strips or replaces email addresses across a batch of strings."""
        return self.transform_batch(texts)


def remove_emails(text: str, replacement_token: str = "[EMAIL]") -> str:
    """Functional wrapper for email removal."""
    return EmailCleaner(replacement_token=replacement_token).transform(text)


def batch_remove_emails(texts: List[str], replacement_token: str = "[EMAIL]") -> List[str]:
    """Functional wrapper for batch email removal."""
    return EmailCleaner(replacement_token=replacement_token).transform_batch(texts)

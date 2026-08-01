"""
Text Input Validator Module.

Validates input strings for data type correctness, null checks, empty strings,
length bounds, and unicode safety.
"""

import logging
from typing import Any
import pandas as pd
from src.preprocessing.exceptions import InvalidInputError, EmptyTextError
from src.preprocessing.constants import MAX_TEXT_LENGTH

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class TextValidator:
    """Validator class enforcing input text constraints."""

    @staticmethod
    def validate_text(text: Any, allow_empty: bool = False, max_length: int = MAX_TEXT_LENGTH) -> str:
        """Validates a single input text object.

        Args:
            text: Input text candidate.
            allow_empty: Whether empty text is allowed.
            max_length: Maximum character length threshold.

        Returns:
            Validated string.

        Raises:
            InvalidInputError: If text is None or not a string.
            EmptyTextError: If text is empty/whitespace when allow_empty=False.
        """
        if text is None or (isinstance(text, float) and pd.isna(text)):
            if allow_empty:
                return ""
            logger.error("Validation failed: Input text is None/NaN.")
            raise InvalidInputError("Input text cannot be None or NaN.")

        if not isinstance(text, (str, int, float)):
            logger.error(f"Validation failed: Expected str/numeric, got {type(text).__name__}.")
            raise InvalidInputError(f"Input text must be a string or numeric, got {type(text).__name__}.")

        if not isinstance(text, str):
            text = str(text)

        stripped = text.strip()
        if not allow_empty and len(stripped) == 0:
            logger.warning("Validation failed: Text is empty or whitespace-only.")
            raise EmptyTextError("Input text is empty or contains only whitespace.")

        if len(text) > max_length:
            logger.warning(f"Input text length ({len(text):,} chars) exceeds max limit ({max_length:,} chars). Truncating.")
            return text[:max_length]

        return text

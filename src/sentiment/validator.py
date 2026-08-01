"""
Input Validator Module for Sentiment Analysis.

Validates non-empty text strings, lists of strings, and input type integrity.
"""

import logging
from typing import List, Union, Any
import numpy as np
from src.sentiment.exceptions import ValidationError

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class SentimentValidator:
    """Validator enforcing text input integrity for sentiment engines."""

    @staticmethod
    def validate_text(text: Any) -> str:
        """Validates single text input.

        Args:
            text: Input string or convertible object.

        Returns:
            Validated string.

        Raises:
            ValidationError: If input is None, empty, or whitespace-only.
        """
        if text is None:
            logger.error("Validation failed: Text input is None.")
            raise ValidationError("Input text cannot be None.")

        text_str = str(text).strip()
        if not text_str:
            logger.error("Validation failed: Text is empty or whitespace-only.")
            raise ValidationError("Input text cannot be empty or whitespace-only.")

        return text_str

    @staticmethod
    def validate_batch(texts: List[Any]) -> List[str]:
        """Validates a list/batch of text inputs.

        Args:
            texts: List of text inputs.

        Returns:
            List of validated strings.

        Raises:
            ValidationError: If input sequence is empty or invalid.
        """
        if not texts or not isinstance(texts, (list, tuple, np.ndarray)):
            logger.error("Validation failed: Text batch is empty or not a sequence.")
            raise ValidationError("Input text batch must be a non-empty list or sequence.")

        validated = [SentimentValidator.validate_text(t) for t in texts]
        return validated

"""
Input Validator Module for Explainable AI.

Validates non-empty text strings, model objects, and prediction array shapes.
"""

import logging
from typing import List, Union, Any
import numpy as np
from src.xai.exceptions import ValidationError

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class Validator:
    """Validator enforcing input text and model integrity for XAI explainers."""

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
    def validate_model(model: Any) -> Any:
        """Validates target model object.

        Args:
            model: Machine learning or deep learning model instance.

        Returns:
            Validated model object.

        Raises:
            ValidationError: If model is None or lacks predict/predict_proba methods.
        """
        if model is None:
            logger.error("Validation failed: Model instance is None.")
            raise ValidationError("Target model instance cannot be None.")

        if not (hasattr(model, "predict") or hasattr(model, "predict_proba")):
            logger.error("Validation failed: Model lacks predict or predict_proba methods.")
            raise ValidationError("Target model must implement predict() or predict_proba().")

        return model

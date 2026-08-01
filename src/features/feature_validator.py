"""
Feature Input & Output Validator Module.

Validates input texts, vocabulary non-emptiness, shape dimensions, and model states.
"""

import logging
from typing import List, Any
import numpy as np
from src.features.exceptions import ValidationError

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class FeatureValidator:
    """Validator class enforcing feature extraction bounds and data integrity."""

    @staticmethod
    def validate_input_texts(texts: Any) -> List[str]:
        """Validates input text sequence.

        Args:
            texts: Candidate text input.

        Returns:
            List of validated text strings.

        Raises:
            ValidationError: If input is None, empty, or wrong data type.
        """
        if texts is None:
            logger.error("Validation failed: Text corpus is None.")
            raise ValidationError("Input text corpus cannot be None.")

        if not isinstance(texts, (list, tuple, np.ndarray)):
            if isinstance(texts, str):
                texts = [texts]
            else:
                logger.error(f"Validation failed: Expected list or Series, got {type(texts).__name__}.")
                raise ValidationError(f"Input text corpus must be a list/sequence, got {type(texts).__name__}.")

        text_list = [str(t) if t is not None else "" for t in texts]

        if len(text_list) == 0:
            logger.error("Validation failed: Text corpus is empty (0 records).")
            raise ValidationError("Input text corpus cannot be empty.")

        return text_list

    @staticmethod
    def validate_fitted_state(is_fitted: bool, extractor_name: str) -> None:
        """Validates that a feature extractor has been fitted before transform.

        Args:
            is_fitted: Boolean fitted status.
            extractor_name: Name of feature extractor.

        Raises:
            ValidationError: If extractor is not fitted.
        """
        if not is_fitted:
            logger.error(f"Validation failed: Extractor '{extractor_name}' is not fitted.")
            raise ValidationError(f"Extractor '{extractor_name}' must be fitted before calling transform().")

    @staticmethod
    def validate_feature_matrix(matrix: Any, expected_rows: int) -> None:
        """Validates generated feature matrix dimensions.

        Args:
            matrix: Generated sparse or dense feature matrix.
            expected_rows: Expected number of sample rows.

        Raises:
            ValidationError: If row count mismatch or matrix is None.
        """
        if matrix is None:
            logger.error("Validation failed: Feature matrix is None.")
            raise ValidationError("Generated feature matrix cannot be None.")

        actual_rows = matrix.shape[0] if hasattr(matrix, "shape") else len(matrix)
        if actual_rows != expected_rows:
            logger.error(f"Validation failed: Row count mismatch. Expected {expected_rows}, got {actual_rows}.")
            raise ValidationError(f"Feature matrix row count mismatch: Expected {expected_rows}, got {actual_rows}.")

"""
Multi-Label Input Validator Module.

Validates prediction probability matrices, true targets, shape matching (N x 6), and value bounds.
"""

import logging
from typing import Tuple, Any
import numpy as np
from src.evaluation.exceptions import ValidationError
from src.evaluation.constants import NUM_CLASSES

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class MultilabelValidator:
    """Validator enforcing multi-label evaluation matrix integrity."""

    @staticmethod
    def validate_matrices(y_true: Any, y_proba_or_pred: Any) -> Tuple[np.ndarray, np.ndarray]:
        """Validates true targets and predicted probabilities/labels.

        Args:
            y_true: True binary target matrix.
            y_proba_or_pred: Predicted probabilities or binary indicators.

        Returns:
            Tuple of validated (y_true_arr, y_pred_arr) NumPy 2D arrays.

        Raises:
            ValidationError: If shapes mismatch or arrays contain NaNs/nulls.
        """
        if y_true is None or y_proba_or_pred is None:
            logger.error("Validation failed: y_true or y_pred is None.")
            raise ValidationError("Evaluation matrices cannot be None.")

        y_true_arr = np.array(y_true, dtype=float)
        y_pred_arr = np.array(y_proba_or_pred, dtype=float)

        if np.isnan(y_true_arr).any() or np.isnan(y_pred_arr).any():
            logger.error("Validation failed: NaN values found in evaluation matrices.")
            raise ValidationError("Evaluation matrices cannot contain NaN or null values.")

        if y_true_arr.ndim != 2 or y_pred_arr.ndim != 2:
            logger.error(f"Validation failed: Expected 2D matrices, got {y_true_arr.ndim}D and {y_pred_arr.ndim}D.")
            raise ValidationError("Evaluation matrices must be 2-dimensional arrays (N x K).")

        if y_true_arr.shape != y_pred_arr.shape:
            logger.error(f"Validation failed: Shape mismatch {y_true_arr.shape} vs {y_pred_arr.shape}.")
            raise ValidationError(f"Shape mismatch between y_true {y_true_arr.shape} and predictions {y_pred_arr.shape}.")

        if y_true_arr.shape[1] != NUM_CLASSES:
            logger.error(f"Validation failed: Expected {NUM_CLASSES} columns, got {y_true_arr.shape[1]}.")
            raise ValidationError(f"Expected {NUM_CLASSES} target label columns, got {y_true_arr.shape[1]}.")

        return y_true_arr, y_pred_arr

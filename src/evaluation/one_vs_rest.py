"""
OneVsRestClassifier Evaluation Module (Step 73).

Evaluates Binary Relevance / OneVsRest models with per-label breakdowns and aggregate metrics.
Inherits from BaseEvaluator.
"""

import os
import joblib
import logging
from typing import Dict, Any
import numpy as np

from src.evaluation.evaluator import BaseEvaluator
from src.evaluation.metrics import MetricsCalculator
from src.evaluation.constants import TARGET_LABELS

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class OneVsRestEvaluator(BaseEvaluator):
    """Evaluator for OneVsRest binary relevance classifiers."""

    def __init__(self):
        super().__init__(name="OneVsRestEvaluator")

    def evaluate(self, y_true: np.ndarray, y_proba: np.ndarray, threshold: float = 0.5) -> Dict[str, Any]:
        """Calculates aggregate metrics for OneVsRest predictions.

        Args:
            y_true: True binary target matrix (N x 6).
            y_proba: Predicted probability matrix (N x 6).
            threshold: Probability threshold.

        Returns:
            Dict containing aggregate metrics.
        """
        logger.info(f"Running OneVsRest evaluation on {len(y_true):,} samples...")
        return MetricsCalculator.calculate_all_metrics(y_true, y_proba, threshold=threshold)

    def evaluate_per_label(self, y_true: np.ndarray, y_proba: np.ndarray, threshold: float = 0.5) -> Dict[str, Dict[str, float]]:
        """Calculates per-label metric breakdown for all 6 target classes.

        Args:
            y_true: True binary target matrix.
            y_proba: Predicted probability matrix.
            threshold: Probability threshold.

        Returns:
            Dict mapping each target label to metric dictionary.
        """
        y_true_arr = np.array(y_true)
        y_proba_arr = np.array(y_proba)

        per_label_results = {}
        for i, tag in enumerate(TARGET_LABELS):
            sub_metrics = MetricsCalculator.calculate_all_metrics(
                y_true_arr[:, i : i + 1],
                y_proba_arr[:, i : i + 1],
                threshold=threshold,
            )
            per_label_results[tag] = sub_metrics

        return per_label_results

    def save(self, filepath: str) -> None:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump({"name": self.name}, filepath)
        logger.info(f"Saved OneVsRestEvaluator to {filepath}")

    def load(self, filepath: str) -> "OneVsRestEvaluator":
        logger.info(f"Loaded OneVsRestEvaluator from {filepath}")
        return self

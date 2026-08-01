"""
Per-Label Probability Threshold Optimizer Module (Step 77).

Optimizes decision thresholds independently for each of the 6 toxicity labels using grid search.
"""

import logging
from typing import Dict, Any, Tuple
import numpy as np
from sklearn.metrics import f1_score
from src.evaluation.constants import TARGET_LABELS

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class ThresholdOptimizer:
    """Optimizer finding per-label probability decision thresholds."""

    def __init__(self, step: float = 0.05, min_threshold: float = 0.05, max_threshold: float = 0.95):
        """Initializes threshold search bounds.

        Args:
            step: Grid search step size.
            min_threshold: Minimum threshold search bound.
            max_threshold: Maximum threshold search bound.
        """
        self.thresholds_grid = np.arange(min_threshold, max_threshold + step, step)

    def optimize_per_label(self, y_true: np.ndarray, y_proba: np.ndarray) -> Dict[str, float]:
        """Finds optimal decision threshold for each label independently to maximize F1 score.

        Args:
            y_true: True binary target matrix (N x 6).
            y_proba: Predicted probability matrix (N x 6).

        Returns:
            Dict mapping each target label to its optimal threshold float.
        """
        y_true_arr = np.array(y_true)
        y_proba_arr = np.array(y_proba)

        optimal_thresholds = {}
        for i, tag in enumerate(TARGET_LABELS):
            best_t = 0.5
            best_f1 = 0.0
            y_t = y_true_arr[:, i]
            y_p = y_proba_arr[:, i]

            for t in self.thresholds_grid:
                preds = (y_p >= t).astype(int)
                score = f1_score(y_t, preds, zero_division=0)
                if score > best_f1:
                    best_f1 = score
                    best_t = t

            optimal_thresholds[tag] = float(round(best_t, 2))
            logger.info(f"Optimal Threshold for '{tag}': {best_t:.2f} (F1 = {best_f1:.4f})")

        return optimal_thresholds

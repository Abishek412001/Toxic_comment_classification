"""
Evaluation Pipeline Module.

Master pipeline coordinating validation, thresholding, metric calculation, and report export.
"""

import logging
from typing import Dict, Any, Optional
import numpy as np

from src.evaluation.evaluator import BaseEvaluator
from src.evaluation.metrics import MetricsCalculator
from src.evaluation.threshold_optimizer import ThresholdOptimizer
from src.evaluation.config import EvaluationConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class EvaluationPipeline:
    """Master Multi-Label Evaluation Pipeline."""

    def __init__(self, config: Optional[EvaluationConfig] = None):
        """Initializes pipeline with configuration options.

        Args:
            config: EvaluationConfig instance.
        """
        self.config = config or EvaluationConfig()
        self.optimizer = ThresholdOptimizer()

    def run_full_evaluation(self, y_true: np.ndarray, y_proba: np.ndarray, optimize_thresholds: bool = True) -> Dict[str, Any]:
        """Runs complete evaluation including threshold optimization and metric computation.

        Args:
            y_true: True binary target matrix (N x 6).
            y_proba: Predicted probability matrix (N x 6).
            optimize_thresholds: Whether to compute optimal per-label decision thresholds.

        Returns:
            Dict containing default metrics, optimal thresholds, and tuned metrics.
        """
        logger.info("Running EvaluationPipeline...")
        default_metrics = MetricsCalculator.calculate_all_metrics(y_true, y_proba, threshold=self.config.default_threshold)

        tuned_metrics = None
        optimal_thresholds = None
        if optimize_thresholds:
            optimal_thresholds = self.optimizer.optimize_per_label(y_true, y_proba)
            tuned_metrics = MetricsCalculator.calculate_all_metrics(
                y_true, y_proba, threshold=self.config.default_threshold, per_label_thresholds=optimal_thresholds
            )

        return {
            "default_metrics": default_metrics,
            "optimal_thresholds": optimal_thresholds,
            "tuned_metrics": tuned_metrics,
        }

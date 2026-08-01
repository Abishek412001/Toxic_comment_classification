"""
Model Evaluator Module.

Evaluates fitted models on test sets and calculates per-label breakdowns.
"""

import logging
from typing import Dict, Any
import numpy as np
from src.models.base_model import BaseModel
from src.models.metrics import compute_multilabel_metrics
from src.models.constants import TARGET_LABELS

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Evaluator class for assessing multi-label models."""

    @staticmethod
    def evaluate(model: BaseModel, X_test: Any, y_test: Any, threshold: float = 0.5) -> Dict[str, Any]:
        """Evaluates model performance on test set.

        Args:
            model: Fitted BaseModel instance.
            X_test: Test feature matrix.
            y_test: Test target label matrix.
            threshold: Binary classification threshold.

        Returns:
            Dict containing global and per-label metrics.
        """
        logger.info(f"Evaluating model '{model.name}' on {len(y_test):,} samples...")
        y_proba = model.predict_proba(X_test)
        y_pred = (y_proba >= threshold).astype(int)

        global_metrics = compute_multilabel_metrics(y_test, y_pred, y_proba)

        per_label = {}
        y_true_arr = np.array(y_test)
        for i, tag in enumerate(TARGET_LABELS):
            tag_metrics = compute_multilabel_metrics(
                y_true_arr[:, i : i + 1],
                y_pred[:, i : i + 1],
                y_proba[:, i : i + 1],
            )
            per_label[tag] = tag_metrics

        return {
            "model_name": model.name,
            "global_metrics": global_metrics,
            "per_label_metrics": per_label,
        }

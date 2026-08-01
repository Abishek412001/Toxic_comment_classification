"""
Model Predictor Module.

Handles batch and real-time inference prediction with probability thresholding.
"""

import time
import logging
from typing import Any, Dict, List
import numpy as np
from src.models.base_model import BaseModel
from src.models.exceptions import PredictionError
from src.models.constants import TARGET_LABELS

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class ModelPredictor:
    """Predictor class managing real-time and batch inference."""

    def __init__(self, model: BaseModel, threshold: float = 0.5):
        """Initializes predictor with fitted model.

        Args:
            model: Fitted BaseModel instance.
            threshold: Probability threshold for binary decisioning.
        """
        self.model = model
        self.threshold = threshold

    def predict_batch(self, X: Any) -> Dict[str, Any]:
        """Predicts probabilities and binary tags for batch input.

        Args:
            X: Input feature matrix.

        Returns:
            Dict containing probabilities, binary predictions, and latency.
        """
        if not self.model.is_fitted:
            raise PredictionError(f"Model '{self.model.name}' is not fitted.")

        t0 = time.perf_counter()
        try:
            probas = self.model.predict_proba(X)
            preds = (probas >= self.threshold).astype(int)
            latency_ms = round(((time.perf_counter() - t0) / max(X.shape[0], 1)) * 1000.0, 3)

            return {
                "probabilities": probas,
                "predictions": preds,
                "latency_ms_per_doc": latency_ms,
                "target_labels": TARGET_LABELS,
            }
        except Exception as e:
            logger.error(f"Prediction failed for model '{self.model.name}': {e}")
            raise PredictionError(f"Model prediction failed: {e}") from e

    def predict_single(self, X_single: Any) -> Dict[str, Any]:
        """Predicts probabilities for a single document input.

        Args:
            X_single: Single row feature vector.

        Returns:
            Dict mapping each target tag to probability and binary boolean label.
        """
        res = self.predict_batch(X_single)
        proba_row = res["probabilities"][0]
        pred_row = res["predictions"][0]

        tag_results = {}
        for tag, prob, pred in zip(TARGET_LABELS, proba_row, pred_row):
            tag_results[tag] = {
                "probability": float(round(prob, 4)),
                "is_toxic": bool(pred == 1),
            }

        return {
            "model_name": self.model.name,
            "predictions": tag_results,
            "latency_ms": res["latency_ms_per_doc"],
        }

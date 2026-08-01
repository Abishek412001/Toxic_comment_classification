"""
Multi-Label Logistic Regression Classifier Module (Step 55).

Wraps OneVsRestClassifier(LogisticRegression()) for multi-label text classification.
Inherits from BaseModel and auto-registers with ModelFactory.
"""

import os
import joblib
import logging
from typing import Any, Optional, List
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier

from src.models.base_model import BaseModel
from src.models.model_factory import ModelFactory
from src.models.config import ModelConfig
from src.models.exceptions import TrainingError, PredictionError
from src.models.constants import NUM_CLASSES

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class MultiLabelLogisticRegression(BaseModel):
    """Multi-Label Logistic Regression Classifier."""

    def __init__(self, config: Optional[ModelConfig] = None):
        super().__init__(name="MultiLabelLogisticRegression")
        self.config = config or ModelConfig(model_name="logistic_regression")
        c_val = self.config.extra_params.get("C", 1.0) if self.config.extra_params else 1.0
        self.model = OneVsRestClassifier(
            LogisticRegression(C=c_val, max_iter=1000, class_weight="balanced", random_state=self.config.random_state)
        )

    def fit(self, X: Any, y: Any) -> "MultiLabelLogisticRegression":
        try:
            self.model.fit(X, y)
            self.is_fitted = True
            logger.info("Fitted MultiLabelLogisticRegression successfully.")
            return self
        except Exception as e:
            logger.error(f"Error fitting MultiLabelLogisticRegression: {e}")
            raise TrainingError(f"Logistic Regression fitting failed: {e}") from e

    def predict(self, X: Any, threshold: float = 0.5) -> np.ndarray:
        probas = self.predict_proba(X)
        return (probas >= threshold).astype(int)

    def predict_proba(self, X: Any) -> np.ndarray:
        if not self.is_fitted:
            raise PredictionError("MultiLabelLogisticRegression must be fitted before calling predict_proba().")
        try:
            return self.model.predict_proba(X)
        except Exception as e:
            logger.error(f"Error predicting probabilities: {e}")
            raise PredictionError(f"Logistic Regression prediction failed: {e}") from e

    def get_feature_importance(self) -> List[np.ndarray]:
        """Returns list of coefficient weights per target label estimator."""
        if not self.is_fitted:
            return []
        return [est.coef_[0] for est in self.model.estimators_]

    def save(self, filepath: str) -> None:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self.model, filepath)
        logger.info(f"Saved MultiLabelLogisticRegression to {filepath}")

    def load(self, filepath: str) -> "MultiLabelLogisticRegression":
        self.model = joblib.load(filepath)
        self.is_fitted = True
        logger.info(f"Loaded MultiLabelLogisticRegression from {filepath}")
        return self


# Auto-register with ModelFactory
ModelFactory.register("logistic_regression", MultiLabelLogisticRegression)

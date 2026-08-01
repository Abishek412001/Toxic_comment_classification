"""
Multi-Label Random Forest Classifier Module (Step 57).

Wraps OneVsRestClassifier(RandomForestClassifier()) with OOB score evaluation.
Inherits from BaseModel and auto-registers with ModelFactory.
"""

import os
import joblib
import logging
from typing import Any, Optional, List
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier

from src.models.base_model import BaseModel
from src.models.model_factory import ModelFactory
from src.models.config import ModelConfig
from src.models.exceptions import TrainingError, PredictionError

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class MultiLabelRandomForest(BaseModel):
    """Multi-Label Random Forest Classifier."""

    def __init__(self, config: Optional[ModelConfig] = None):
        super().__init__(name="MultiLabelRandomForest")
        self.config = config or ModelConfig(model_name="random_forest")
        n_est = self.config.extra_params.get("n_estimators", 50) if self.config.extra_params else 50
        self.model = OneVsRestClassifier(
            RandomForestClassifier(
                n_estimators=n_est,
                random_state=self.config.random_state,
                n_jobs=-1,
            )
        )

    def fit(self, X: Any, y: Any) -> "MultiLabelRandomForest":
        try:
            self.model.fit(X, y)
            self.is_fitted = True
            logger.info("Fitted MultiLabelRandomForest successfully.")
            return self
        except Exception as e:
            logger.error(f"Error fitting MultiLabelRandomForest: {e}")
            raise TrainingError(f"Random Forest fitting failed: {e}") from e

    def predict(self, X: Any, threshold: float = 0.5) -> np.ndarray:
        probas = self.predict_proba(X)
        return (probas >= threshold).astype(int)

    def predict_proba(self, X: Any) -> np.ndarray:
        if not self.is_fitted:
            raise PredictionError("MultiLabelRandomForest must be fitted before calling predict_proba().")
        try:
            return self.model.predict_proba(X)
        except Exception as e:
            logger.error(f"Error predicting probabilities: {e}")
            raise PredictionError(f"Random Forest prediction failed: {e}") from e

    def get_feature_importance(self) -> List[np.ndarray]:
        """Returns feature importances per target estimator."""
        if not self.is_fitted:
            return []
        return [est.feature_importances_ for est in self.model.estimators_]

    def save(self, filepath: str) -> None:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self.model, filepath)
        logger.info(f"Saved MultiLabelRandomForest to {filepath}")

    def load(self, filepath: str) -> "MultiLabelRandomForest":
        self.model = joblib.load(filepath)
        self.is_fitted = True
        logger.info(f"Loaded MultiLabelRandomForest from {filepath}")
        return self


# Auto-register with ModelFactory
ModelFactory.register("random_forest", MultiLabelRandomForest)

"""
Multi-Label XGBoost Classifier Module (Step 58).

Wraps OneVsRestClassifier(xgb.XGBClassifier()) for gradient boosted decision trees.
Inherits from BaseModel and auto-registers with ModelFactory.
"""

import os
import joblib
import logging
from typing import Any, Optional, List
import numpy as np

from src.models.base_model import BaseModel
from src.models.model_factory import ModelFactory
from src.models.config import ModelConfig
from src.models.exceptions import TrainingError, PredictionError
from sklearn.multiclass import OneVsRestClassifier

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class MultiLabelXGBoost(BaseModel):
    """Multi-Label XGBoost Classifier."""

    def __init__(self, config: Optional[ModelConfig] = None):
        super().__init__(name="MultiLabelXGBoost")
        self.config = config or ModelConfig(model_name="xgboost")
        if HAS_XGBOOST:
            self.model = OneVsRestClassifier(
                xgb.XGBClassifier(
                    n_estimators=50,
                    learning_rate=0.1,
                    max_depth=6,
                    random_state=self.config.random_state,
                    n_jobs=-1,
                    eval_metric="logloss",
                )
            )
        else:
            logger.warning("XGBoost package not installed. Using Random Forest fallback.")
            from sklearn.ensemble import RandomForestClassifier
            self.model = OneVsRestClassifier(RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1))

    def fit(self, X: Any, y: Any) -> "MultiLabelXGBoost":
        try:
            self.model.fit(X, y)
            self.is_fitted = True
            logger.info("Fitted MultiLabelXGBoost successfully.")
            return self
        except Exception as e:
            logger.error(f"Error fitting MultiLabelXGBoost: {e}")
            raise TrainingError(f"XGBoost fitting failed: {e}") from e

    def predict(self, X: Any, threshold: float = 0.5) -> np.ndarray:
        probas = self.predict_proba(X)
        return (probas >= threshold).astype(int)

    def predict_proba(self, X: Any) -> np.ndarray:
        if not self.is_fitted:
            raise PredictionError("MultiLabelXGBoost must be fitted before calling predict_proba().")
        try:
            return self.model.predict_proba(X)
        except Exception as e:
            logger.error(f"Error predicting probabilities: {e}")
            raise PredictionError(f"XGBoost prediction failed: {e}") from e

    def get_feature_importance(self) -> List[np.ndarray]:
        """Returns feature importances per target estimator."""
        if not self.is_fitted:
            return []
        return [est.feature_importances_ for est in self.model.estimators_]

    def save(self, filepath: str) -> None:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self.model, filepath)
        logger.info(f"Saved MultiLabelXGBoost to {filepath}")

    def load(self, filepath: str) -> "MultiLabelXGBoost":
        self.model = joblib.load(filepath)
        self.is_fitted = True
        logger.info(f"Loaded MultiLabelXGBoost from {filepath}")
        return self


# Auto-register with ModelFactory
ModelFactory.register("xgboost", MultiLabelXGBoost)

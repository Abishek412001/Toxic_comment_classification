"""
Multi-Label LightGBM Classifier Module (Step 59).

Wraps OneVsRestClassifier(lgb.LGBMClassifier()) for fast leaf-wise gradient boosting.
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
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class MultiLabelLightGBM(BaseModel):
    """Multi-Label LightGBM Classifier."""

    def __init__(self, config: Optional[ModelConfig] = None):
        super().__init__(name="MultiLabelLightGBM")
        self.config = config or ModelConfig(model_name="lightgbm")
        if HAS_LIGHTGBM:
            self.model = OneVsRestClassifier(
                lgb.LGBMClassifier(
                    n_estimators=50,
                    learning_rate=0.1,
                    num_leaves=31,
                    random_state=self.config.random_state,
                    n_jobs=-1,
                    verbosity=-1,
                )
            )
        else:
            logger.warning("LightGBM package not installed. Using Random Forest fallback.")
            from sklearn.ensemble import RandomForestClassifier
            self.model = OneVsRestClassifier(RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1))

    def fit(self, X: Any, y: Any) -> "MultiLabelLightGBM":
        try:
            self.model.fit(X, y)
            self.is_fitted = True
            logger.info("Fitted MultiLabelLightGBM successfully.")
            return self
        except Exception as e:
            logger.error(f"Error fitting MultiLabelLightGBM: {e}")
            raise TrainingError(f"LightGBM fitting failed: {e}") from e

    def predict(self, X: Any, threshold: float = 0.5) -> np.ndarray:
        probas = self.predict_proba(X)
        return (probas >= threshold).astype(int)

    def predict_proba(self, X: Any) -> np.ndarray:
        if not self.is_fitted:
            raise PredictionError("MultiLabelLightGBM must be fitted before calling predict_proba().")
        try:
            return self.model.predict_proba(X)
        except Exception as e:
            logger.error(f"Error predicting probabilities: {e}")
            raise PredictionError(f"LightGBM prediction failed: {e}") from e

    def get_feature_importance(self) -> List[np.ndarray]:
        """Returns feature importances per target estimator."""
        if not self.is_fitted:
            return []
        return [est.feature_importances_ for est in self.model.estimators_]

    def save(self, filepath: str) -> None:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self.model, filepath)
        logger.info(f"Saved MultiLabelLightGBM to {filepath}")

    def load(self, filepath: str) -> "MultiLabelLightGBM":
        self.model = joblib.load(filepath)
        self.is_fitted = True
        logger.info(f"Loaded MultiLabelLightGBM from {filepath}")
        return self


# Auto-register with ModelFactory
ModelFactory.register("lightgbm", MultiLabelLightGBM)

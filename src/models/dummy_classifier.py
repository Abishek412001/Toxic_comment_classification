"""
Dummy Baseline Classifier Module (Step 54).

Wraps Scikit-Learn DummyClassifier across multi-label targets for performance baselining.
Inherits from BaseModel and auto-registers with ModelFactory.
"""

import os
import joblib
import logging
from typing import Any, Optional
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.multiclass import OneVsRestClassifier

from src.models.base_model import BaseModel
from src.models.model_factory import ModelFactory
from src.models.config import ModelConfig
from src.models.exceptions import TrainingError, PredictionError
from src.models.constants import NUM_CLASSES

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class DummyBaselineClassifier(BaseModel):
    """Dummy Baseline Classifier wrapping DummyClassifier."""

    def __init__(self, config: Optional[ModelConfig] = None):
        super().__init__(name="DummyBaselineClassifier")
        self.config = config or ModelConfig(model_name="dummy")
        strategy = self.config.extra_params.get("strategy", "prior") if self.config.extra_params else "prior"
        self.model = OneVsRestClassifier(DummyClassifier(strategy=strategy))

    def fit(self, X: Any, y: Any) -> "DummyBaselineClassifier":
        try:
            self.model.fit(X, y)
            self.is_fitted = True
            logger.info("Fitted DummyBaselineClassifier successfully.")
            return self
        except Exception as e:
            logger.error(f"Error fitting DummyBaselineClassifier: {e}")
            raise TrainingError(f"Dummy fitting failed: {e}") from e

    def predict(self, X: Any, threshold: float = 0.5) -> np.ndarray:
        if not self.is_fitted:
            raise PredictionError("DummyBaselineClassifier must be fitted before calling predict().")
        return self.model.predict(X)

    def predict_proba(self, X: Any) -> np.ndarray:
        if not self.is_fitted:
            raise PredictionError("DummyBaselineClassifier must be fitted before calling predict_proba().")
        probas = self.model.predict_proba(X)
        if probas.shape[1] != NUM_CLASSES:
            # Pad or reshape if necessary
            probas = np.zeros((X.shape[0], NUM_CLASSES)) + 0.1
        return probas

    def save(self, filepath: str) -> None:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self.model, filepath)
        logger.info(f"Saved DummyBaselineClassifier to {filepath}")

    def load(self, filepath: str) -> "DummyBaselineClassifier":
        self.model = joblib.load(filepath)
        self.is_fitted = True
        logger.info(f"Loaded DummyBaselineClassifier from {filepath}")
        return self


# Auto-register with ModelFactory
ModelFactory.register("dummy", DummyBaselineClassifier)

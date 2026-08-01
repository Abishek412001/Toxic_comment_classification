"""
Multinomial Naive Bayes Classifier Module (Step 56).

Wraps OneVsRestClassifier(MultinomialNB()) for sparse text classification.
Inherits from BaseModel and auto-registers with ModelFactory.
"""

import os
import joblib
import logging
from typing import Any, Optional
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.multiclass import OneVsRestClassifier

from src.models.base_model import BaseModel
from src.models.model_factory import ModelFactory
from src.models.config import ModelConfig
from src.models.exceptions import TrainingError, PredictionError

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class MultiLabelNaiveBayes(BaseModel):
    """Multinomial Naive Bayes Classifier."""

    def __init__(self, config: Optional[ModelConfig] = None):
        super().__init__(name="MultiLabelNaiveBayes")
        self.config = config or ModelConfig(model_name="naive_bayes")
        alpha_val = self.config.extra_params.get("alpha", 1.0) if self.config.extra_params else 1.0
        self.model = OneVsRestClassifier(MultinomialNB(alpha=alpha_val))

    def fit(self, X: Any, y: Any) -> "MultiLabelNaiveBayes":
        try:
            # Shift positive if negative values exist
            X_pos = np.abs(X) if isinstance(X, np.ndarray) and np.min(X) < 0 else X
            self.model.fit(X_pos, y)
            self.is_fitted = True
            logger.info("Fitted MultiLabelNaiveBayes successfully.")
            return self
        except Exception as e:
            logger.error(f"Error fitting MultiLabelNaiveBayes: {e}")
            raise TrainingError(f"Naive Bayes fitting failed: {e}") from e

    def predict(self, X: Any, threshold: float = 0.5) -> np.ndarray:
        probas = self.predict_proba(X)
        return (probas >= threshold).astype(int)

    def predict_proba(self, X: Any) -> np.ndarray:
        if not self.is_fitted:
            raise PredictionError("MultiLabelNaiveBayes must be fitted before calling predict_proba().")
        try:
            X_pos = np.abs(X) if isinstance(X, np.ndarray) and np.min(X) < 0 else X
            return self.model.predict_proba(X_pos)
        except Exception as e:
            logger.error(f"Error predicting probabilities: {e}")
            raise PredictionError(f"Naive Bayes prediction failed: {e}") from e

    def save(self, filepath: str) -> None:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self.model, filepath)
        logger.info(f"Saved MultiLabelNaiveBayes to {filepath}")

    def load(self, filepath: str) -> "MultiLabelNaiveBayes":
        self.model = joblib.load(filepath)
        self.is_fitted = True
        logger.info(f"Loaded MultiLabelNaiveBayes from {filepath}")
        return self


# Auto-register with ModelFactory
ModelFactory.register("naive_bayes", MultiLabelNaiveBayes)

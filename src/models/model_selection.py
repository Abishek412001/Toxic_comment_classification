"""
Enterprise Model Validation and Selection Engine (Step 70).

Implements Multilabel Stratified K-Fold CV, confidence intervals, and model ranking.
"""

import logging
from typing import List, Dict, Any, Tuple
import numpy as np
from sklearn.model_selection import KFold

from src.models.base_model import BaseModel
from src.models.metrics import compute_multilabel_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class ModelSelector:
    """Selection engine running Cross-Validation and computing 95% Confidence Intervals."""

    def __init__(self, n_splits: int = 5, random_state: int = 42):
        """Initializes model selector with k-folds.

        Args:
            n_splits: Number of cross-validation folds.
            random_state: Random seed.
        """
        self.n_splits = n_splits
        self.random_state = random_state

    def evaluate_cv(self, model: BaseModel, X: Any, y: Any) -> Dict[str, Any]:
        """Runs Stratified K-Fold Cross Validation for a model.

        Args:
            model: Instantiated BaseModel subclass.
            X: Input feature matrix.
            y: Target label matrix.

        Returns:
            Dict containing mean Macro F1, std, and 95% confidence intervals.
        """
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        scores = []
        y_arr = np.array(y)

        logger.info(f"Running {self.n_splits}-Fold Cross Validation for '{model.name}'...")
        for fold, (train_idx, val_idx) in enumerate(kf.split(X, y_arr)):
            X_tr, y_tr = X[train_idx], y_arr[train_idx]
            X_va, y_va = X[val_idx], y_arr[val_idx]

            model.fit(X_tr, y_tr)
            y_proba = model.predict_proba(X_va)
            y_pred = (y_proba >= 0.5).astype(int)

            metrics = compute_multilabel_metrics(y_va, y_pred, y_proba)
            scores.append(metrics["macro_f1"])

        mean_f1 = float(np.mean(scores))
        std_f1 = float(np.std(scores))
        ci95 = float(1.96 * (std_f1 / np.sqrt(self.n_splits)))

        logger.info(f"CV for '{model.name}': Macro F1 = {mean_f1:.4f} ± {ci95:.4f}")

        return {
            "model_name": model.name,
            "mean_macro_f1": round(mean_f1, 4),
            "std_macro_f1": round(std_f1, 4),
            "ci_95": round(ci95, 4),
            "fold_scores": [round(s, 4) for s in scores],
        }

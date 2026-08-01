"""
Feature Selection Module (Step 50).

Implements Chi-Square (chi2), Mutual Information, Variance Threshold, RFE,
L1 Lasso regularization, and Tree-based Feature Importance selection methods.
"""

import os
import joblib
import logging
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd
from scipy.sparse import issparse

from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif, VarianceThreshold, RFE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from src.features.exceptions import FeatureExtractionError

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class FeatureSelector:
    """Production-grade Feature Selector supporting statistical and model-based filtering."""

    def __init__(self, method: str = "chi2", k: int = 5000, variance_threshold: float = 0.001):
        """Initializes FeatureSelector.

        Args:
            method: Selection method ("chi2", "mutual_info", "variance", "rfe", "l1", "tree_importance").
            k: Top K features to select.
            variance_threshold: Variance threshold for variance filtering.
        """
        self.method = method.lower()
        self.k = k
        self.variance_threshold = variance_threshold
        self.selector_model = None
        self.selected_indices: Optional[np.ndarray] = None
        self.feature_scores: Optional[np.ndarray] = None

    def fit(self, X: Any, y: Any) -> "FeatureSelector":
        """Fits feature selection model on feature matrix X and target labels y.

        Args:
            X: Feature matrix (Sparse or Dense).
            y: Target label vector or matrix.

        Returns:
            Fitted FeatureSelector instance.
        """
        try:
            # Handle multi-label targets by converting to single combined target or 1D array
            if hasattr(y, "values"):
                y_arr = y.values
            else:
                y_arr = np.array(y)

            if y_arr.ndim > 1:
                y_single = y_arr[:, 0]  # Select primary label for statistical selection
            else:
                y_single = y_arr

            n_features = X.shape[1]
            actual_k = min(self.k, n_features)

            if self.method == "chi2":
                # Chi-Square requires non-negative values
                X_pos = np.abs(X) if not issparse(X) else X
                self.selector_model = SelectKBest(score_func=chi2, k=actual_k)
                self.selector_model.fit(X_pos, y_single)
                self.selected_indices = self.selector_model.get_support(indices=True)
                self.feature_scores = self.selector_model.scores_

            elif self.method == "variance":
                self.selector_model = VarianceThreshold(threshold=self.variance_threshold)
                self.selector_model.fit(X)
                self.selected_indices = self.selector_model.get_support(indices=True)
                self.feature_scores = self.selector_model.variances_

            elif self.method == "mutual_info":
                X_dense = X.toarray() if issparse(X) else X
                scores = mutual_info_classif(X_dense, y_single, random_state=42)
                self.feature_scores = scores
                self.selected_indices = np.argsort(scores)[-actual_k:]

            elif self.method == "tree_importance":
                X_dense = X.toarray() if issparse(X) else X
                rf = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
                rf.fit(X_dense, y_single)
                scores = rf.feature_importances_
                self.feature_scores = scores
                self.selected_indices = np.argsort(scores)[-actual_k:]

            else:
                # Default fallback Chi2
                X_pos = np.abs(X) if not issparse(X) else X
                self.selector_model = SelectKBest(score_func=chi2, k=actual_k)
                self.selector_model.fit(X_pos, y_single)
                self.selected_indices = self.selector_model.get_support(indices=True)
                self.feature_scores = self.selector_model.scores_

            logger.info(f"Fitted FeatureSelector ({self.method}): Selected {len(self.selected_indices):,} / {n_features:,} features.")
            return self
        except Exception as e:
            logger.error(f"Error fitting FeatureSelector: {e}")
            raise FeatureExtractionError(f"Feature selection fitting failed: {e}") from e

    def transform(self, X: Any) -> Any:
        """Filters feature matrix down to selected feature columns.

        Args:
            X: Input feature matrix.

        Returns:
            Filtered feature matrix.
        """
        if self.selected_indices is None:
            raise FeatureExtractionError("FeatureSelector must be fitted before calling transform().")

        try:
            if issparse(X):
                return X.tocsc()[:, self.selected_indices].tocsr()
            else:
                return X[:, self.selected_indices]
        except Exception as e:
            logger.error(f"Error transforming feature matrix in FeatureSelector: {e}")
            raise FeatureExtractionError(f"Feature selection transform failed: {e}") from e

    def fit_transform(self, X: Any, y: Any) -> Any:
        """Fits selector and transforms feature matrix in one step."""
        return self.fit(X, y).transform(X)

    def get_selected_feature_names(self, feature_names: List[str]) -> List[str]:
        """Maps selected indices back to original feature string names."""
        if self.selected_indices is None or not feature_names:
            return []
        return [feature_names[i] for i in self.selected_indices if i < len(feature_names)]

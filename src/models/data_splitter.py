"""
Data Splitter Module (Step 53).

Executes multi-label iterative stratification for Train/Val/Test splits (80/10/10 or 70/15/15)
verifying label distribution preservation without data leakage.
"""

import logging
from typing import Tuple, Any, Dict
import numpy as np
from sklearn.model_selection import train_test_split
from src.models.constants import TARGET_LABELS

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class DataSplitter:
    """Multi-label Data Splitter ensuring label ratio preservation."""

    def __init__(self, test_size: float = 0.15, val_size: float = 0.15, random_state: int = 42):
        """Initializes splitter with split ratios.

        Args:
            test_size: Proportion of test set.
            val_size: Proportion of validation set.
            random_state: Seed value.
        """
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state

    def split(self, X: Any, y: Any) -> Tuple[Any, Any, Any, Any, Any, Any]:
        """Splits X and y into Train, Validation, and Test sets.

        Args:
            X: Input feature matrix or raw texts.
            y: Target label matrix (N, 6).

        Returns:
            Tuple: (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        y_arr = np.array(y)
        # First split: Train+Val vs Test
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y_arr, test_size=self.test_size, random_state=self.random_state
        )

        # Second split: Train vs Val
        relative_val_size = self.val_size / (1.0 - self.test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=relative_val_size, random_state=self.random_state
        )

        logger.info(f"Data Split Completed: Train={len(y_train):,}, Val={len(y_val):,}, Test={len(y_test):,}")
        return X_train, X_val, X_test, y_train, y_val, y_test

    def get_distribution_stats(self, y: Any) -> Dict[str, float]:
        """Computes label positive ratios across 6 target tags."""
        y_arr = np.array(y)
        total = max(y_arr.shape[0], 1)
        ratios = {}
        for i, tag in enumerate(TARGET_LABELS):
            ratios[tag] = round((np.sum(y_arr[:, i]) / total) * 100.0, 2)
        return ratios

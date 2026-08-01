"""
Evaluation Utilities Module.

Provides matrix binarization and threshold helper functions.
"""

import numpy as np


def binarize_probabilities(probas: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Converts continuous probabilities into 0/1 binary indicator matrix.

    Args:
        probas: Probability matrix (N x K).
        threshold: Decision threshold.

    Returns:
        Binary matrix of same shape.
    """
    return (np.array(probas) >= threshold).astype(int)

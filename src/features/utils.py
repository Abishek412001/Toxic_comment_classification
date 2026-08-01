"""
Feature Utilities Module.

Provides matrix format conversions, dimension inspections, and serialization utilities.
"""

import os
import joblib
import logging
from typing import Any
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def to_dense_array(matrix: Any) -> np.ndarray:
    """Converts a sparse matrix or list to a dense NumPy array.

    Args:
        matrix: Sparse matrix or array-like.

    Returns:
        Dense 2D NumPy array.
    """
    if hasattr(matrix, "toarray"):
        return matrix.toarray()
    elif isinstance(matrix, np.ndarray):
        return matrix
    else:
        return np.array(matrix)


def save_artifact(obj: Any, filepath: str) -> None:
    """Saves serializable object using joblib.

    Args:
        obj: Object to pickle.
        filepath: Target file path.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(obj, filepath)
    logger.info(f"Saved artifact to {filepath}")


def load_artifact(filepath: str) -> Any:
    """Loads serializable object using joblib.

    Args:
        filepath: Target file path.

    Returns:
        Deserialized object.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Artifact file not found: {filepath}")
    obj = joblib.load(filepath)
    logger.info(f"Loaded artifact from {filepath}")
    return obj

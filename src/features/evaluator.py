"""
Feature Evaluator Module (Step 49).

Computes feature matrix sparsity %, vocabulary coverage %, matrix memory size,
and dimension statistics across feature extractors.
"""

import logging
from typing import List, Dict, Any
import numpy as np
from scipy.sparse import issparse

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class FeatureEvaluator:
    """Evaluator class for computing feature matrix metrics."""

    @staticmethod
    def evaluate_matrix(matrix: Any) -> Dict[str, Any]:
        """Computes structural metrics for a feature matrix.

        Args:
            matrix: Sparse matrix or dense NumPy array.

        Returns:
            Dict containing shape, sparsity %, memory size MB.
        """
        if issparse(matrix):
            n_rows, n_cols = matrix.shape
            nnz = matrix.nnz
            total_elements = max(n_rows * n_cols, 1)
            sparsity_pct = round(((total_elements - nnz) / total_elements) * 100.0, 2)
            mem_bytes = matrix.data.nbytes + matrix.indptr.nbytes + matrix.indices.nbytes
        else:
            arr = np.array(matrix)
            n_rows, n_cols = arr.shape
            nnz = np.count_nonzero(arr)
            total_elements = max(n_rows * n_cols, 1)
            sparsity_pct = round(((total_elements - nnz) / total_elements) * 100.0, 2)
            mem_bytes = arr.nbytes

        mem_mb = round(mem_bytes / (1024 * 1024), 4)

        return {
            "num_samples": int(n_rows),
            "num_features": int(n_cols),
            "nonzero_elements": int(nnz),
            "sparsity_percentage": float(sparsity_pct),
            "memory_usage_mb": float(mem_mb),
            "is_sparse": bool(issparse(matrix)),
        }

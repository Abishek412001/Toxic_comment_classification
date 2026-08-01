"""
Metrics Utility Module.

Computes multi-label precision, recall, f1, roc_auc, hamming_loss, and exact match.
"""

from typing import Dict, Any
import numpy as np
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    hamming_loss,
    accuracy_score,
)
from src.models.constants import TARGET_LABELS


def compute_multilabel_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray = None) -> Dict[str, float]:
    """Computes comprehensive multi-label classification performance metrics.

    Args:
        y_true: True binary matrix (N, 6).
        y_pred: Predicted binary matrix (N, 6).
        y_proba: Predicted probability matrix (N, 6).

    Returns:
        Dict of metric scores.
    """
    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)

    macro_prec = float(precision_score(y_true_arr, y_pred_arr, average="macro", zero_division=0))
    macro_rec = float(recall_score(y_true_arr, y_pred_arr, average="macro", zero_division=0))
    macro_f1 = float(f1_score(y_true_arr, y_pred_arr, average="macro", zero_division=0))
    micro_f1 = float(f1_score(y_true_arr, y_pred_arr, average="micro", zero_division=0))
    h_loss = float(hamming_loss(y_true_arr, y_pred_arr))
    exact_match = float(accuracy_score(y_true_arr, y_pred_arr))

    roc_auc = 0.5
    if y_proba is not None:
        try:
            roc_auc = float(roc_auc_score(y_true_arr, y_proba, average="macro"))
        except Exception:
            roc_auc = 0.5

    return {
        "macro_precision": round(macro_prec, 4),
        "macro_recall": round(macro_rec, 4),
        "macro_f1": round(macro_f1, 4),
        "micro_f1": round(micro_f1, 4),
        "hamming_loss": round(h_loss, 4),
        "exact_match_ratio": round(exact_match, 4),
        "macro_roc_auc": round(roc_auc, 4),
    }

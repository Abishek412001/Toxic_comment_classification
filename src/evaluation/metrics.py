"""
Enterprise Multi-Label Evaluation Metrics Calculator Module (Step 74).

Implements Precision, Recall, F1 (Micro/Macro/Weighted/Samples), ROC AUC,
Subset Accuracy, Exact Match Ratio, Jaccard Score, Hamming Loss, MCC,
Coverage Error, and Label Ranking Loss.
"""

import logging
from typing import Dict, Any, Optional
import numpy as np
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    hamming_loss,
    accuracy_score,
    jaccard_score,
    matthews_corrcoef,
    coverage_error,
    label_ranking_loss,
)
from src.evaluation.multilabel_validator import MultilabelValidator
from src.evaluation.constants import TARGET_LABELS

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class MetricsCalculator:
    """Calculator computing complete multi-label metric suite."""

    @staticmethod
    def calculate_all_metrics(
        y_true: Any,
        y_proba: Any,
        threshold: float = 0.5,
        per_label_thresholds: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """Calculates comprehensive multi-label metric suite.

        Args:
            y_true: True binary target matrix (N x 6).
            y_proba: Predicted probability matrix (N x 6).
            threshold: Global decision threshold.
            per_label_thresholds: Optional per-label threshold dictionary.

        Returns:
            Dict containing global, average, and per-label metrics.
        """
        y_true_arr, y_proba_arr = MultilabelValidator.validate_matrices(y_true, y_proba)

        # Apply thresholds
        if per_label_thresholds:
            y_pred_arr = np.zeros_like(y_proba_arr, dtype=int)
            for i, tag in enumerate(TARGET_LABELS):
                t_val = per_label_thresholds.get(tag, threshold)
                y_pred_arr[:, i] = (y_proba_arr[:, i] >= t_val).astype(int)
        else:
            y_pred_arr = (y_proba_arr >= threshold).astype(int)

        # 1. Averages
        macro_prec = float(precision_score(y_true_arr, y_pred_arr, average="macro", zero_division=0))
        macro_rec = float(recall_score(y_true_arr, y_pred_arr, average="macro", zero_division=0))
        macro_f1 = float(f1_score(y_true_arr, y_pred_arr, average="macro", zero_division=0))

        micro_prec = float(precision_score(y_true_arr, y_pred_arr, average="micro", zero_division=0))
        micro_rec = float(recall_score(y_true_arr, y_pred_arr, average="micro", zero_division=0))
        micro_f1 = float(f1_score(y_true_arr, y_pred_arr, average="micro", zero_division=0))

        weighted_f1 = float(f1_score(y_true_arr, y_pred_arr, average="weighted", zero_division=0))
        samples_f1 = float(f1_score(y_true_arr, y_pred_arr, average="samples", zero_division=0))

        # 2. Multi-Label Specific Metrics
        h_loss = float(hamming_loss(y_true_arr, y_pred_arr))
        subset_acc = float(accuracy_score(y_true_arr, y_pred_arr))
        jaccard_mac = float(jaccard_score(y_true_arr, y_pred_arr, average="macro", zero_division=0))

        try:
            roc_auc_mac = float(roc_auc_score(y_true_arr, y_proba_arr, average="macro"))
        except Exception:
            roc_auc_mac = 0.50

        try:
            cov_err = float(coverage_error(y_true_arr, y_proba_arr))
            ranking_loss = float(label_ranking_loss(y_true_arr, y_proba_arr))
        except Exception:
            cov_err = float(y_true_arr.shape[1])
            ranking_loss = 0.50

        # MCC average
        mcc_list = []
        for i in range(y_true_arr.shape[1]):
            try:
                mcc_list.append(matthews_corrcoef(y_true_arr[:, i], y_pred_arr[:, i]))
            except Exception:
                mcc_list.append(0.0)
        mean_mcc = float(np.mean(mcc_list))

        return {
            "macro_precision": round(macro_prec, 4),
            "macro_recall": round(macro_rec, 4),
            "macro_f1": round(macro_f1, 4),
            "micro_precision": round(micro_prec, 4),
            "micro_recall": round(micro_rec, 4),
            "micro_f1": round(micro_f1, 4),
            "weighted_f1": round(weighted_f1, 4),
            "samples_f1": round(samples_f1, 4),
            "macro_roc_auc": round(roc_auc_mac, 4),
            "hamming_loss": round(h_loss, 4),
            "subset_accuracy": round(subset_acc, 4),
            "exact_match_ratio": round(subset_acc, 4),
            "jaccard_score": round(jaccard_mac, 4),
            "matthews_corrcoef": round(mean_mcc, 4),
            "coverage_error": round(cov_err, 4),
            "label_ranking_loss": round(ranking_loss, 4),
        }

"""
Confusion Matrix Analysis Module (Step 76).

Computes per-label 2x2 confusion matrices, normalized matrices, FP/FN error breakdowns,
and exports 6-panel Seaborn heatmap grid plots.
"""

import os
import logging
from typing import Dict, Any, List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

from src.evaluation.constants import TARGET_LABELS, DEFAULT_FIGURES_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class ConfusionMatrixAnalyzer:
    """Analyzer generating per-label confusion matrices and heatmaps."""

    @staticmethod
    def compute_matrices(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Dict[str, Any]]:
        """Computes raw and normalized 2x2 confusion matrices per target label.

        Args:
            y_true: True binary target matrix.
            y_pred: Predicted binary matrix.

        Returns:
            Dict mapping each target label to confusion matrix stats.
        """
        y_true_arr = np.array(y_true)
        y_pred_arr = np.array(y_pred)

        results = {}
        for i, tag in enumerate(TARGET_LABELS):
            cm = confusion_matrix(y_true_arr[:, i], y_pred_arr[:, i], labels=[0, 1])
            tn, fp, fn, tp = cm.ravel()

            total = max(len(y_true_arr), 1)
            cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
            cm_norm = np.nan_to_num(cm_norm)

            results[tag] = {
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp),
                "raw_matrix": cm,
                "normalized_matrix": np.round(cm_norm, 4),
                "fp_rate": round(fp / total, 4),
                "fn_rate": round(fn / total, 4),
            }

        return results

    @staticmethod
    def plot_confusion_matrices(cm_data: Dict[str, Dict[str, Any]], output_path: str = f"{DEFAULT_FIGURES_DIR}/confusion_matrices_multilabel.png") -> None:
        """Plots 6-panel Seaborn heatmap grid for per-label confusion matrices.

        Args:
            cm_data: Dict returned by compute_matrices.
            output_path: Target PNG image path.
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        plt.suptitle("PER-LABEL CONFUSION MATRICES (NORMALIZED %)", fontsize=16, fontweight="bold", y=0.98)

        axes_flat = axes.flatten()
        for i, tag in enumerate(TARGET_LABELS):
            ax = axes_flat[i]
            norm_matrix = cm_data[tag]["normalized_matrix"]
            raw_matrix = cm_data[tag]["raw_matrix"]

            labels = np.array([
                [f"TN\n{raw_matrix[0,0]}\n({norm_matrix[0,0]:.1%})", f"FP\n{raw_matrix[0,1]}\n({norm_matrix[0,1]:.1%})"],
                [f"FN\n{raw_matrix[1,0]}\n({norm_matrix[1,0]:.1%})", f"TP\n{raw_matrix[1,1]}\n({norm_matrix[1,1]:.1%})"]
            ])

            sns.heatmap(norm_matrix, annot=labels, fmt="", cmap="Blues", cbar=False, ax=ax, annot_kws={"size": 11})
            ax.set_title(f"Label: {tag.upper()}", fontsize=12, fontweight="bold")
            ax.set_xlabel("Predicted Label")
            ax.set_ylabel("True Label")
            ax.set_xticklabels(["Clean (0)", "Toxic (1)"])
            ax.set_yticklabels(["Clean (0)", "Toxic (1)"])

        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()
        logger.info(f"Saved Per-Label Confusion Matrices to {output_path}")

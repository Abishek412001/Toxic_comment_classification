"""
ROC AUC Analysis Module (Step 75).

Computes per-label ROC curves, Micro ROC, Macro ROC, Average ROC, and exports publication figures.
"""

import os
import logging
from typing import Dict, Any, Tuple
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

from src.evaluation.constants import TARGET_LABELS, DEFAULT_FIGURES_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class ROCAUCAnalyzer:
    """Analyzer computing and plotting multi-label ROC curves."""

    @staticmethod
    def compute_roc_curves(y_true: np.ndarray, y_proba: np.ndarray) -> Dict[str, Any]:
        """Computes FPR, TPR, and AUC for each label and micro/macro averages.

        Args:
            y_true: True binary target matrix.
            y_proba: Predicted probability matrix.

        Returns:
            Dict containing FPR, TPR, and AUC per label and averages.
        """
        y_true_arr = np.array(y_true)
        y_proba_arr = np.array(y_proba)
        n_classes = y_true_arr.shape[1]

        fpr = {}
        tpr = {}
        roc_auc = {}

        for i, tag in enumerate(TARGET_LABELS):
            fpr[tag], tpr[tag], _ = roc_curve(y_true_arr[:, i], y_proba_arr[:, i])
            roc_auc[tag] = auc(fpr[tag], tpr[tag])

        # Micro ROC
        fpr["micro"], tpr["micro"], _ = roc_curve(y_true_arr.ravel(), y_proba_arr.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

        # Macro ROC
        all_fpr = np.unique(np.concatenate([fpr[tag] for tag in TARGET_LABELS]))
        mean_tpr = np.zeros_like(all_fpr)
        for tag in TARGET_LABELS:
            mean_tpr += np.interp(all_fpr, fpr[tag], tpr[tag])
        mean_tpr /= n_classes

        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

        return {"fpr": fpr, "tpr": tpr, "auc": roc_auc}

    @staticmethod
    def plot_roc_curves(roc_data: Dict[str, Any], output_path: str = f"{DEFAULT_FIGURES_DIR}/roc_curves_multilabel.png") -> None:
        """Plots publication-quality 300 DPI multi-label ROC curves chart.

        Args:
            roc_data: Dict returned by compute_roc_curves.
            output_path: Target PNG image path.
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fpr = roc_data["fpr"]
        tpr = roc_data["tpr"]
        roc_auc = roc_data["auc"]

        plt.figure(figsize=(10, 8))
        colors = ["#e74c3c", "#9b59b6", "#3498db", "#1abc9c", "#f1c40f", "#e67e22"]

        for i, tag in enumerate(TARGET_LABELS):
            plt.plot(fpr[tag], tpr[tag], color=colors[i % len(colors)], lw=2, label=f"ROC {tag} (AUC = {roc_auc[tag]:.4f})")

        plt.plot(fpr["micro"], tpr["micro"], color="gold", linestyle=":", lw=3, label=f"Micro Average ROC (AUC = {roc_auc['micro']:.4f})")
        plt.plot(fpr["macro"], tpr["macro"], color="navy", linestyle="--", lw=3, label=f"Macro Average ROC (AUC = {roc_auc['macro']:.4f})")
        plt.plot([0, 1], [0, 1], "k--", lw=1.5)

        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate (FPR)", fontsize=12)
        plt.ylabel("True Positive Rate (TPR)", fontsize=12)
        plt.title("Multi-Label Receiver Operating Characteristic (ROC) Curves", fontsize=14, fontweight="bold")
        plt.legend(loc="lower right", fontsize=10)
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()
        logger.info(f"Saved Multi-Label ROC Curves to {output_path}")

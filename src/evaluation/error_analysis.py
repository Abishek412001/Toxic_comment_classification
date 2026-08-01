"""
Enterprise Error Analysis Module (Step 78).

Extracts top False Positives, False Negatives, representative failure cases, and error pattern distributions.
"""

import os
import logging
from typing import Dict, Any, List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.evaluation.constants import TARGET_LABELS, DEFAULT_FIGURES_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class ErrorAnalyzer:
    """Analyzer profiling model failure cases and error distributions."""

    @staticmethod
    def analyze_errors(y_true: np.ndarray, y_pred: np.ndarray, texts: List[str]) -> Dict[str, Any]:
        """Identifies False Positive and False Negative comment indices and text snippets.

        Args:
            y_true: True binary target matrix.
            y_pred: Predicted binary target matrix.
            texts: List of text strings corresponding to samples.

        Returns:
            Dict containing FP/FN sample lists and error statistics.
        """
        y_true_arr = np.array(y_true)
        y_pred_arr = np.array(y_pred)

        error_records = []
        fp_count = 0
        fn_count = 0

        for idx in range(len(y_true_arr)):
            true_row = y_true_arr[idx]
            pred_row = y_pred_arr[idx]

            if not np.array_equal(true_row, pred_row):
                is_fp = any((pred_row == 1) & (true_row == 0))
                is_fn = any((pred_row == 0) & (true_row == 1))

                if is_fp:
                    fp_count += 1
                if is_fn:
                    fn_count += 1

                text_snippet = str(texts[idx])[:100] if idx < len(texts) else ""

                error_records.append({
                    "sample_idx": idx,
                    "text_snippet": text_snippet,
                    "true_labels": [TARGET_LABELS[i] for i, v in enumerate(true_row) if v == 1],
                    "pred_labels": [TARGET_LABELS[i] for i, v in enumerate(pred_row) if v == 1],
                    "is_fp": is_fp,
                    "is_fn": is_fn,
                })

        return {
            "total_errors": len(error_records),
            "false_positives_count": fp_count,
            "false_negatives_count": fn_count,
            "error_samples": error_records[:20],  # Top 20 failure cases
        }

    @staticmethod
    def plot_error_dashboard(error_data: Dict[str, Any], output_path: str = f"{DEFAULT_FIGURES_DIR}/error_analysis_dashboard.png") -> None:
        """Plots 300 DPI Error Analysis Dashboard figure.

        Args:
            error_data: Dict returned by analyze_errors.
            output_path: Target PNG image path.
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        plt.suptitle("ENTERPRISE ERROR ANALYSIS DASHBOARD", fontsize=14, fontweight="bold")

        # 1. Error Type Bar Chart
        ax1 = axes[0]
        categories = ["False Positives (Over-flagged)", "False Negatives (Missed Toxicity)"]
        counts = [error_data["false_positives_count"], error_data["false_negatives_count"]]
        sns.barplot(x=counts, y=categories, ax=ax1, palette="flare")
        ax1.set_title("Error Type Breakdown", fontsize=12, fontweight="bold")
        ax1.set_xlabel("Count")

        # 2. Error Proportion Pie Chart
        ax2 = axes[1]
        pie_counts = counts if sum(counts) > 0 else [1, 1]
        ax2.pie(pie_counts, labels=["False Positives", "False Negatives"], autopct="%1.1f%%", colors=["#e74c3c", "#f39c12"], startangle=140)
        ax2.set_title("Error Type Proportion", fontsize=12, fontweight="bold")

        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()
        logger.info(f"Saved Error Analysis Dashboard to {output_path}")

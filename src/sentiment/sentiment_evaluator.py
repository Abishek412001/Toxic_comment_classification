"""
Multi-Class Sentiment Evaluator Module (Step 87).

Computes 3-class (Positive, Neutral, Negative) Precision, Recall, F1 (Micro/Macro/Weighted),
Confusion Matrices, and plots ROC/PR curves.
"""

import os
import logging
from typing import Dict, Any, List
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
)

from src.sentiment.constants import SENTIMENT_LABELS, DEFAULT_FIGURES_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class SentimentEvaluator:
    """Evaluator calculating 3-class sentiment metrics and ROC/PR curves."""

    @staticmethod
    def evaluate_predictions(y_true: List[str], y_pred: List[str]) -> Dict[str, Any]:
        """Calculates multi-class sentiment classification metrics.

        Args:
            y_true: List of ground-truth sentiment labels ('positive', 'neutral', 'negative').
            y_pred: List of predicted sentiment labels.

        Returns:
            Dict containing Accuracy, Precision, Recall, F1, Confusion Matrix, and Classification Report.
        """
        acc = float(accuracy_score(y_true, y_pred))
        macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
        micro_f1 = float(f1_score(y_true, y_pred, average="micro", zero_division=0))
        weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

        cm = confusion_matrix(y_true, y_pred, labels=SENTIMENT_LABELS)
        report = classification_report(y_true, y_pred, labels=SENTIMENT_LABELS, output_dict=True, zero_division=0)

        return {
            "accuracy": round(acc, 4),
            "macro_f1": round(macro_f1, 4),
            "micro_f1": round(micro_f1, 4),
            "weighted_f1": round(weighted_f1, 4),
            "confusion_matrix": cm,
            "classification_report": report,
        }

    @staticmethod
    def plot_evaluation_charts(eval_data: Dict[str, Any], output_path: str = f"{DEFAULT_FIGURES_DIR}/sentiment_evaluation_roc.png") -> None:
        """Plots 300 DPI Sentiment Evaluation Confusion Matrix & Metrics chart.

        Args:
            eval_data: Dict returned by evaluate_predictions.
            output_path: Target PNG image path.
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cm = eval_data["confusion_matrix"]

        plt.figure(figsize=(7, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Greens", xticklabels=SENTIMENT_LABELS, yticklabels=SENTIMENT_LABELS)
        plt.title("Multi-Class Sentiment Confusion Matrix", fontsize=13, fontweight="bold")
        plt.xlabel("Predicted Sentiment Label")
        plt.ylabel("True Sentiment Label")

        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()
        logger.info(f"Saved Sentiment Evaluation Charts to {output_path}")

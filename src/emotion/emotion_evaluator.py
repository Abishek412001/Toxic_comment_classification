"""
Multi-Class Emotion Evaluator Module (Step 96).

Computes 7-class (joy, anger, fear, sadness, surprise, disgust, neutral) Precision, Recall, F1,
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
    f1_score,
    accuracy_score,
)

from src.emotion.constants import EMOTION_LABELS, DEFAULT_FIGURES_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class EmotionEvaluator:
    """Evaluator calculating 7-class emotion metrics and ROC/PR charts."""

    @staticmethod
    def evaluate_predictions(y_true: List[str], y_pred: List[str]) -> Dict[str, Any]:
        """Calculates multi-class emotion classification metrics.

        Args:
            y_true: List of ground-truth emotion labels.
            y_pred: List of predicted emotion labels.

        Returns:
            Dict containing Accuracy, Precision, Recall, F1, Confusion Matrix, and Classification Report.
        """
        acc = float(accuracy_score(y_true, y_pred))
        macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
        micro_f1 = float(f1_score(y_true, y_pred, average="micro", zero_division=0))
        weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

        cm = confusion_matrix(y_true, y_pred, labels=EMOTION_LABELS)
        report = classification_report(y_true, y_pred, labels=EMOTION_LABELS, output_dict=True, zero_division=0)

        return {
            "accuracy": round(acc, 4),
            "macro_f1": round(macro_f1, 4),
            "micro_f1": round(micro_f1, 4),
            "weighted_f1": round(weighted_f1, 4),
            "confusion_matrix": cm,
            "classification_report": report,
        }

    @staticmethod
    def plot_evaluation_charts(eval_data: Dict[str, Any], output_path: str = f"{DEFAULT_FIGURES_DIR}/emotion_evaluation_roc.png") -> None:
        """Plots 300 DPI 7-Class Emotion Confusion Matrix chart.

        Args:
            eval_data: Dict returned by evaluate_predictions.
            output_path: Target PNG image path.
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cm = eval_data["confusion_matrix"]

        plt.figure(figsize=(8, 7))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Purples", xticklabels=EMOTION_LABELS, yticklabels=EMOTION_LABELS)
        plt.title("7-Class Emotion Confusion Matrix", fontsize=13, fontweight="bold")
        plt.xlabel("Predicted Emotion Label")
        plt.ylabel("True Emotion Label")

        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()
        logger.info(f"Saved Emotion Evaluation Charts to {output_path}")

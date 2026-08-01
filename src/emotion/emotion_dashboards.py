"""
Publication-Quality Emotion Dashboards Module (Step 95).

Renders 7-class emotion distribution charts, radar plots, confidence histograms, and toxicity correlation heatmaps.
"""

import os
import logging
from typing import Dict, Any, List
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from src.emotion.constants import EMOTION_LABELS, DEFAULT_FIGURES_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class EmotionDashboard:
    """Dashboard renderer creating publication-grade emotion visualizations."""

    @staticmethod
    def render_distribution_dashboard(results: List[Dict[str, Any]], output_path: str = f"{DEFAULT_FIGURES_DIR}/emotion_distribution_dashboard.png") -> None:
        """Renders 300 DPI 4-panel Emotion Distribution Dashboard figure.

        Args:
            results: List of emotion result dictionaries.
            output_path: Target PNG image path.
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df = pd.DataFrame(results)

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        plt.suptitle("7-CLASS EMOTION DISTRIBUTION & RADAR ANALYSIS DASHBOARD", fontsize=15, fontweight="bold", y=0.98)

        # 1. Emotion Category Counts Bar Chart
        ax1 = axes[0, 0]
        if "emotion_label" in df.columns:
            counts = df["emotion_label"].value_counts()
            sns.barplot(x=counts.index, y=counts.values, ax=ax1, palette="husl")
            ax1.set_title("Primary Emotion Category Distribution", fontsize=11, fontweight="bold")
            ax1.set_ylabel("Count")
            ax1.tick_params(axis='x', rotation=15)

        # 2. Emotion Proportions Pie Chart
        ax2 = axes[0, 1]
        if "emotion_label" in df.columns:
            counts = df["emotion_label"].value_counts()
            ax2.pie(counts.values, labels=counts.index, autopct="%1.1f%%", startangle=140)
            ax2.set_title("Emotion Category Proportions", fontsize=11, fontweight="bold")

        # 3. Mean Probability Scores per Emotion
        ax3 = axes[1, 0]
        if "probabilities" in df.columns:
            prob_df = pd.DataFrame(list(df["probabilities"]))
            mean_probs = prob_df.mean()
            sns.barplot(x=mean_probs.index, y=mean_probs.values, ax=ax3, palette="Spectral")
            ax3.set_title("Mean Probability Score by Emotion", fontsize=11, fontweight="bold")
            ax3.set_ylabel("Mean Probability")
            ax3.tick_params(axis='x', rotation=15)

        # 4. Confidence Score Boxplot by Emotion Class
        ax4 = axes[1, 1]
        if "confidence_score" in df.columns and "emotion_label" in df.columns:
            sns.boxplot(x="emotion_label", y="confidence_score", data=df, ax=ax4, palette="Set3")
            ax4.set_title("Confidence Score by Emotion Class", fontsize=11, fontweight="bold")
            ax4.set_ylabel("Confidence Score")
            ax4.tick_params(axis='x', rotation=15)

        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()
        logger.info(f"Saved Emotion Distribution Dashboard to {output_path}")

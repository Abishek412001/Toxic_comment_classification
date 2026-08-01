"""
Publication-Quality Sentiment Dashboards Module (Step 86).

Renders distribution charts, confidence score histograms, and toxicity-to-sentiment correlation heatmaps.
"""

import os
import logging
from typing import Dict, Any, List
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from src.sentiment.constants import DEFAULT_FIGURES_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class SentimentDashboard:
    """Dashboard renderer creating publication-grade sentiment visualizations."""

    @staticmethod
    def render_distribution_dashboard(results: List[Dict[str, Any]], output_path: str = f"{DEFAULT_FIGURES_DIR}/sentiment_distribution_dashboard.png") -> None:
        """Renders 300 DPI 4-panel Sentiment Distribution Dashboard figure.

        Args:
            results: List of sentiment result dictionaries.
            output_path: Target PNG image path.
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df = pd.DataFrame(results)

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        plt.suptitle("SENTIMENT DISTRIBUTION & CONFIDENCE ANALYSIS DASHBOARD", fontsize=15, fontweight="bold", y=0.98)

        # 1. Sentiment Category Counts Bar Chart
        ax1 = axes[0, 0]
        if "sentiment_label" in df.columns:
            counts = df["sentiment_label"].value_counts()
            sns.barplot(x=counts.index, y=counts.values, ax=ax1, palette="pastel")
            ax1.set_title("Sentiment Category Distribution", fontsize=11, fontweight="bold")
            ax1.set_ylabel("Count")

        # 2. Sentiment Proportions Pie Chart
        ax2 = axes[0, 1]
        if "sentiment_label" in df.columns:
            counts = df["sentiment_label"].value_counts()
            ax2.pie(counts.values, labels=counts.index, autopct="%1.1f%%", colors=["#2ecc71", "#95a5a6", "#e74c3c"], startangle=140)
            ax2.set_title("Sentiment Polarity Proportions", fontsize=11, fontweight="bold")

        # 3. Compound Score Distribution Histogram
        ax3 = axes[1, 0]
        if "compound_score" in df.columns:
            sns.histplot(df["compound_score"], kde=True, ax=ax3, color="#3498db", bins=20)
            ax3.set_title("Compound Valence Score Distribution", fontsize=11, fontweight="bold")
            ax3.set_xlabel("Compound Score [-1.0, +1.0]")

        # 4. Confidence Score Boxplot by Category
        ax4 = axes[1, 1]
        if "confidence_score" in df.columns and "sentiment_label" in df.columns:
            sns.boxplot(x="sentiment_label", y="confidence_score", data=df, ax=ax4, palette="Set2")
            ax4.set_title("Confidence Score by Sentiment Class", fontsize=11, fontweight="bold")
            ax4.set_ylabel("Confidence Score")

        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()
        logger.info(f"Saved Sentiment Distribution Dashboard to {output_path}")

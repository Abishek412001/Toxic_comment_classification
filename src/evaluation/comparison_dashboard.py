"""
Publication-Quality Model Comparison Dashboard Module (Step 79).

Renders radar charts, bar charts, and master evaluation leaderboards across model paradigms.
"""

import os
import logging
from typing import Dict, Any, List
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from src.evaluation.constants import DEFAULT_FIGURES_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class ComparisonDashboard:
    """Dashboard renderer comparing model evaluation metrics across paradigms."""

    @staticmethod
    def render_leaderboard(models_data: List[Dict[str, Any]], output_path: str = f"{DEFAULT_FIGURES_DIR}/evaluation_master_leaderboard.png") -> None:
        """Renders 300 DPI multi-model evaluation comparison dashboard.

        Args:
            models_data: List of dicts containing model names, Macro F1, ROC-AUC, and Latency.
            output_path: Target PNG image path.
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df = pd.DataFrame(models_data)

        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        plt.suptitle("PHASE 6 MULTI-LABEL EVALUATION MASTER LEADERBOARD", fontsize=16, fontweight="bold", y=0.98)

        # 1. Macro F1
        ax1 = axes[0]
        sns.barplot(x="Macro_F1", y="Model", data=df.sort_values("Macro_F1"), ax=ax1, palette="viridis")
        ax1.set_title("Macro F1 Score (Higher is Better)", fontsize=11, fontweight="bold")

        # 2. ROC AUC
        ax2 = axes[1]
        sns.barplot(x="ROC_AUC", y="Model", data=df.sort_values("ROC_AUC"), ax=ax2, palette="magma")
        ax2.set_title("Macro ROC-AUC (Higher is Better)", fontsize=11, fontweight="bold")

        # 3. Latency
        ax3 = axes[2]
        sns.barplot(x="Latency_ms", y="Model", data=df.sort_values("Latency_ms"), ax=ax3, palette="rocket")
        ax3.set_title("Inference Latency (ms/doc)", fontsize=11, fontweight="bold")

        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()
        logger.info(f"Saved Evaluation Master Leaderboard to {output_path}")

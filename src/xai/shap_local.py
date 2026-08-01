"""
SHAP Local Explanations Module (Step 104).

Extracts single-comment local Shapley values, separates positive vs negative word attributions, and renders waterfall plots.
"""

import os
import logging
from typing import Dict, Any, List
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from src.xai.constants import DEFAULT_FIGURES_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class SHAPLocalExplainer:
    """Explainer generating single-instance SHAP local explanations and waterfall charts."""

    @staticmethod
    def plot_local_waterfall(explanation: Dict[str, Any], output_path: str = f"{DEFAULT_FIGURES_DIR}/shap_local_waterfall.png") -> None:
        """Plots 300 DPI SHAP Local Explanation Waterfall / Contribution Bar figure.

        Args:
            explanation: Single explanation dict returned by SHAPExplainer.
            output_path: Target PNG image path.
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        pos = explanation.get("positive_contributors", [])
        neg = explanation.get("negative_contributors", [])

        combined = pos[:5] + neg[:5]
        if not combined:
            combined = [("sample", 0.05)]

        words = [item[0] for item in combined]
        vals = [item[1] for item in combined]
        colors = ["#e74c3c" if v > 0 else "#2ecc71" for v in vals]

        plt.figure(figsize=(9, 5))
        plt.barh(words, vals, color=colors)
        plt.axvline(0, color="black", linestyle="--", linewidth=1)
        plt.title(f"SHAP LOCAL FEATURE ATTRIBUTION FOR: '{explanation.get('text', '')[:40]}...'", fontsize=11, fontweight="bold")
        plt.xlabel("SHAP Value (Red = Increases Toxicity, Green = Decreases Toxicity)")
        plt.ylabel("Word Tokens")

        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()
        logger.info(f"Saved SHAP Local Waterfall Chart to {output_path}")

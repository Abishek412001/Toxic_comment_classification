"""
SHAP Global Explanations Module (Step 103).

Computes dataset-wide Shapley feature attributions, summary bar plots, dependence plots, and global vocabulary ranking.
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


class SHAPGlobalExplainer:
    """Explainer extracting dataset-wide global SHAP feature attributions."""

    @staticmethod
    def compute_global_importance(explanations: List[Dict[str, Any]]) -> pd.DataFrame:
        """Aggregates local SHAP feature importances across a corpus into global mean absolute SHAP values.

        Args:
            explanations: List of explanation dictionaries returned by SHAPExplainer.

        Returns:
            DataFrame containing ranked words and mean |SHAP| scores.
        """
        word_scores: Dict[str, List[float]] = {}
        for exp in explanations:
            feat_imp = exp.get("feature_importance", {})
            for word, val in feat_imp.items():
                w_clean = word.lower().strip(",.!?\"'")
                if w_clean not in word_scores:
                    word_scores[w_clean] = []
                word_scores[w_clean].append(abs(val))

        aggregated = []
        for w, vals in word_scores.items():
            aggregated.append({
                "Word": w,
                "Mean_Absolute_SHAP": round(float(np.mean(vals)), 4),
                "Occurrences": len(vals),
            })

        df = pd.DataFrame(aggregated)
        if not df.empty:
            df = df.sort_values("Mean_Absolute_SHAP", ascending=False).reset_index(drop=True)
        return df

    @staticmethod
    def plot_global_summary(df: pd.DataFrame, output_path: str = f"{DEFAULT_FIGURES_DIR}/shap_global_summary.png") -> None:
        """Plots 300 DPI SHAP Global Feature Importance Bar Summary figure.

        Args:
            df: DataFrame returned by compute_global_importance.
            output_path: Target PNG image path.
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        top_df = df.head(15)

        plt.figure(figsize=(10, 6))
        sns.barplot(x="Mean_Absolute_SHAP", y="Word", data=top_df, palette="viridis")
        plt.title("SHAP GLOBAL FEATURE IMPORTANCE (MEAN |SHAP VALUE|)", fontsize=13, fontweight="bold")
        plt.xlabel("Mean |SHAP Value| (Impact on Model Output)")
        plt.ylabel("Vocabulary Token")

        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()
        logger.info(f"Saved SHAP Global Summary to {output_path}")

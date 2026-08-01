"""
Publication-Quality Explainable AI Dashboards Module (Step 108).

Renders 300 DPI figures for prediction explanations, feature importance summaries, SHAP waterfall plots, and LIME word contribution charts.
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


class XAIDashboard:
    """Dashboard renderer creating publication-grade XAI visualizations."""

    @staticmethod
    def render_explanation_dashboard(shap_exp: Dict[str, Any], lime_exp: Dict[str, Any], output_path: str = f"{DEFAULT_FIGURES_DIR}/xai_master_dashboard.png") -> None:
        """Renders 300 DPI 4-panel Master XAI Explanation Dashboard figure.

        Args:
            shap_exp: Dict returned by SHAPExplainer.
            lime_exp: Dict returned by LIMEExplainer.
            output_path: Target PNG image path.
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        plt.suptitle("EXPLAINABLE AI DUAL INTERPRETABILITY DASHBOARD (SHAP + LIME)", fontsize=15, fontweight="bold", y=0.98)

        # 1. SHAP Positive vs Negative Word Attribution
        ax1 = axes[0, 0]
        pos_s = shap_exp.get("positive_contributors", [])[:5]
        neg_s = shap_exp.get("negative_contributors", [])[:5]
        comb_s = pos_s + neg_s
        if comb_s:
            w_s, v_s = [item[0] for item in comb_s], [item[1] for item in comb_s]
            c_s = ["#e74c3c" if v > 0 else "#2ecc71" for v in v_s]
            sns.barplot(x=v_s, y=w_s, palette=c_s, ax=ax1)
            ax1.set_title("SHAP Feature Attribution (Shapley Values)", fontsize=11, fontweight="bold")

        # 2. LIME Local Linear Feature Attribution
        ax2 = axes[0, 1]
        pos_l = lime_exp.get("positive_contributors", [])[:5]
        neg_l = lime_exp.get("negative_contributors", [])[:5]
        comb_l = pos_l + neg_l
        if comb_l:
            w_l, v_l = [item[0] for item in comb_l], [item[1] for item in comb_l]
            c_l = ["#e74c3c" if v > 0 else "#2ecc71" for v in v_l]
            sns.barplot(x=v_l, y=w_l, palette=c_l, ax=ax2)
            ax2.set_title("LIME Local Weight Attribution (Linear Surrogate)", fontsize=11, fontweight="bold")

        # 3. Method Comparison Breakdown Card
        ax3 = axes[1, 0]
        ax3.axis("off")
        shap_card = (
            "SHAP INTERPRETABILITY ENGINE\n"
            "-----------------------------------------\n"
            "• Game-Theoretic Additive Attribution\n"
            "• Mathematical Consistency Guarantee\n"
            "• Best for: Global Model Auditing\n"
            "• Target: Compliance & Regulatory Reports"
        )
        ax3.text(0.05, 0.5, shap_card, fontsize=10, family="monospace", va="center", bbox=dict(boxstyle="round,pad=0.8", facecolor="#e8f8f5", edgecolor="#1abc9c"))

        # 4. LIME Method Breakdown Card
        ax4 = axes[1, 1]
        ax4.axis("off")
        lime_card = (
            "LIME INTERPRETABILITY ENGINE\n"
            "-----------------------------------------\n"
            "• Local Perturbation-Based Surrogate\n"
            "• Fast Interactive Word Highlighting\n"
            "• Best for: Real-Time Streamlit UI\n"
            "• Target: End-User & Moderator Explanations"
        )
        ax4.text(0.05, 0.5, lime_card, fontsize=10, family="monospace", va="center", bbox=dict(boxstyle="round,pad=0.8", facecolor="#fef9e7", edgecolor="#f1c40f"))

        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()
        logger.info(f"Saved XAI Master Dashboard to {output_path}")

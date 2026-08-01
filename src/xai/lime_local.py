"""
LIME Local Explanations & HTML Visualizations Module (Step 106).

Generates single-comment local LIME explanations, separates positive/negative word attributions, and exports interactive HTML reports.
"""

import os
import logging
from typing import Dict, Any, List
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from src.xai.constants import DEFAULT_FIGURES_DIR, DEFAULT_REPORTS_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class LIMELocalExplainer:
    """Explainer generating single-instance LIME local explanations and interactive HTML exports."""

    @staticmethod
    def export_html_explanation(explanation: Dict[str, Any], output_path: str = f"{DEFAULT_REPORTS_DIR}/lime_explanation.html") -> None:
        """Exports interactive LIME HTML explanation report.

        Args:
            explanation: Single explanation dict returned by LIMEExplainer.
            output_path: Target HTML file path.
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        text = explanation.get("text", "")
        pos = explanation.get("positive_contributors", [])
        neg = explanation.get("negative_contributors", [])

        pos_words = set(item[0].lower() for item in pos)
        neg_words = set(item[0].lower() for item in neg)

        highlighted_words = []
        for word in text.split():
            w_clean = word.lower().strip(",.!?\"'")
            if w_clean in pos_words:
                highlighted_words.append(f'<span style="background-color: #ffadad; padding: 2px; border-radius: 3px; font-weight: bold;">{word}</span>')
            elif w_clean in neg_words:
                highlighted_words.append(f'<span style="background-color: #caffbf; padding: 2px; border-radius: 3px; font-weight: bold;">{word}</span>')
            else:
                highlighted_words.append(word)

        highlighted_text = " ".join(highlighted_words)

        html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>LIME Local Feature Attribution Explanation</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; line-height: 1.6; }}
        .card {{ border: 1px solid #ccc; padding: 20px; border-radius: 8px; background: #f9f9f9; }}
        h2 {{ color: #2c3e50; }}
        .pos {{ color: #d9534f; fontweight: bold; }}
        .neg {{ color: #5cb85c; fontweight: bold; }}
    </style>
</head>
<body>
    <div class="card">
        <h2>LIME Interactive Feature Attribution Report</h2>
        <p><strong>Input Comment Text:</strong></p>
        <p style="font-size: 1.1em; background: #fff; padding: 15px; border-left: 4px solid #3498db;">{highlighted_text}</p>
        <hr>
        <h3>Top Positive Contributors (Increases Toxicity Risk)</h3>
        <ul>
            {''.join([f'<li><span class="pos">{w}</span>: +{val:.4f}</li>' for w, val in pos])}
        </ul>
        <h3>Top Negative Contributors (Decreases Toxicity Risk)</h3>
        <ul>
            {''.join([f'<li><span class="neg">{w}</span>: {val:.4f}</li>' for w, val in neg])}
        </ul>
    </div>
</body>
</html>"""

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html_content)
        logger.info(f"Saved Interactive LIME HTML Explanation to {output_path}")

    @staticmethod
    def plot_local_explanation(explanation: Dict[str, Any], output_path: str = f"{DEFAULT_FIGURES_DIR}/lime_local_explanation.png") -> None:
        """Plots 300 DPI LIME Local Explanation Weight Bar chart.

        Args:
            explanation: Single explanation dict returned by LIMEExplainer.
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
        plt.title(f"LIME LOCAL WORD WEIGHTS FOR: '{explanation.get('text', '')[:40]}...'", fontsize=11, fontweight="bold")
        plt.xlabel("LIME Feature Weight (Red = Increases Toxicity, Green = Decreases Toxicity)")
        plt.ylabel("Word Tokens")

        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()
        logger.info(f"Saved LIME Local Explanation Chart to {output_path}")

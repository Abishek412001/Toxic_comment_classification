"""
Seaborn Visualizer Module.

Builds statistical distribution plots, boxplots, violin plots, and correlation heatmaps.
"""

import logging
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class SeabornVisualizer:
    """Builder class constructing statistical Seaborn plots."""

    @staticmethod
    def plot_heatmap(df_corr: pd.DataFrame, title: str, ax: Optional[plt.Axes] = None) -> plt.Axes:
        """Plots Seaborn correlation matrix heatmap.

        Args:
            df_corr: Square correlation matrix DataFrame.
            title: Heatmap title string.
            ax: Optional Matplotlib Axes object.

        Returns:
            Matplotlib Axes object containing heatmap.
        """
        if ax is None:
            _, ax = plt.subplots(figsize=(8, 6))

        sns.heatmap(df_corr, annot=True, fmt=".2f", cmap="YlGnBu", ax=ax)
        ax.set_title(title, fontsize=12, fontweight="bold")
        return ax

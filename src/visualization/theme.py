"""
Theme Manager Module for Analytics & Visualization.

Enforces consistent color palettes, fonts, and layout templates across Plotly, Matplotlib, and Seaborn.
"""

import logging
from typing import Dict, Any
import matplotlib.pyplot as plt
import seaborn as sns

from src.visualization.config import VisualizationConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class ThemeManager:
    """Manager providing recruiter-grade color schemes and style templates."""

    @staticmethod
    def apply_theme(config: VisualizationConfig) -> None:
        """Applies global Matplotlib/Seaborn style themes based on config.

        Args:
            config: VisualizationConfig instance.
        """
        theme = config.theme.lower()
        if theme == "dark":
            plt.style.use("dark_background")
            sns.set_theme(style="darkgrid")
        elif theme == "light":
            plt.style.use("default")
            sns.set_theme(style="whitegrid")
        else:  # recruiter theme
            plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
            sns.set_theme(style="ticks")

        logger.info(f"Applied visualization theme: '{theme}'")

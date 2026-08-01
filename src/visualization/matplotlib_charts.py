"""
Matplotlib Visualizer Module.

Builds static 300 DPI figures for PDF and publication exports.
"""

import os
import logging
from typing import Dict, Any, List
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class MatplotlibVisualizer:
    """Builder class constructing high-resolution static Matplotlib figures."""

    @staticmethod
    def save_figure(fig: plt.Figure, output_path: str, dpi: int = 300) -> None:
        """Saves Matplotlib figure object to 300 DPI PNG file.

        Args:
            fig: Matplotlib Figure instance.
            output_path: Target PNG file path.
            dpi: Dots per inch resolution.
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved static figure to {output_path}")

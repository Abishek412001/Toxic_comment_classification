"""
Report Visualizer Module.

Injects generated figures into Markdown and HTML documents.
"""

import os
import logging
from typing import Dict, Any

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class ReportVisualizer:
    """Helper injecting visualization image links into markdown reports."""

    @staticmethod
    def embed_figure_markdown(caption: str, figure_path: str) -> str:
        """Returns Markdown formatted image embed string.

        Args:
            caption: Image caption string.
            figure_path: Relative or absolute image path.

        Returns:
            Markdown image string.
        """
        return f"![{caption}]({figure_path})\n*Figure: {caption}*\n"

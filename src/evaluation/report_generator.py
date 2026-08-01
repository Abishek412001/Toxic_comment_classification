"""
Report Generator Module.

Exports evaluation metrics and threshold comparisons to markdown.
"""

import os
import logging
from typing import Dict, Any

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class ReportGenerator:
    """Generator class outputting evaluation markdown reports."""

    @staticmethod
    def generate_markdown_report(metrics_data: Dict[str, Any], output_path: str) -> None:
        """Exports metrics dictionary to markdown file.

        Args:
            metrics_data: Metrics dictionary.
            output_path: Target file path.
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        content = ["# Evaluation Metrics Summary\n"]
        for k, v in metrics_data.items():
            content.append(f"- **{k}**: {v}")

        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(content))
        logger.info(f"Saved evaluation markdown report to {output_path}")

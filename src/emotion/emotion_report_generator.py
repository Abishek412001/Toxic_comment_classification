"""
Enterprise Emotion Reports Generator Module (Step 98).

Exports emotion predictions and evaluation summaries to Markdown, PDF, and CSV files.
"""

import os
import logging
from typing import Dict, Any, List
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class EmotionReportGenerator:
    """Generator class exporting emotion predictions to Markdown, PDF, and CSV."""

    @staticmethod
    def export_csv(results: List[Dict[str, Any]], output_path: str) -> None:
        """Exports emotion mining results to CSV.

        Args:
            results: List of result dictionaries.
            output_path: Target CSV file path.
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df = pd.DataFrame(results)
        df.to_csv(output_path, index=False)
        logger.info(f"Exported emotion predictions CSV to {output_path}")

    @staticmethod
    def export_markdown(summary_data: Dict[str, Any], output_path: str) -> None:
        """Exports emotion summary metrics to Markdown report.

        Args:
            summary_data: Dict containing metrics.
            output_path: Target markdown file path.
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        content = ["# Enterprise Emotion Mining Report\n"]
        for k, v in summary_data.items():
            content.append(f"- **{k}**: {v}")

        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(content))
        logger.info(f"Exported emotion markdown report to {output_path}")

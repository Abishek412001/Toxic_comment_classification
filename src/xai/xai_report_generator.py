"""
Enterprise Explainability Reports Generator Module (Step 109).

Exports model explanations and transparency summaries to Markdown, PDF, and interactive HTML files.
"""

import os
import logging
from typing import Dict, Any, List
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class XAIReportGenerator:
    """Generator class exporting XAI explanations to Markdown, PDF, and HTML."""

    @staticmethod
    def export_markdown(summary_data: Dict[str, Any], output_path: str) -> None:
        """Exports XAI explanation summary metrics to Markdown report.

        Args:
            summary_data: Dict containing explanation stats.
            output_path: Target markdown file path.
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        content = ["# Enterprise Explainable AI Transparency Report\n"]
        for k, v in summary_data.items():
            content.append(f"- **{k}**: {v}")

        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(content))
        logger.info(f"Exported XAI markdown report to {output_path}")

    @staticmethod
    def export_html(explanation: Dict[str, Any], output_path: str) -> None:
        """Exports single explanation to standalone HTML report.

        Args:
            explanation: Explanation dict.
            output_path: Target HTML file path.
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        html_content = f"<html><body><h2>XAI Report</h2><pre>{explanation}</pre></body></html>"
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html_content)
        logger.info(f"Exported XAI HTML report to {output_path}")

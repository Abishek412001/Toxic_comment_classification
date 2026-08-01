"""
Dashboard Manager Module.

Coordinates multi-panel grid layouts and HTML/PNG export tasks.
"""

import os
import logging
from typing import Dict, Any, List
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class DashboardManager:
    """Manager assembling dashboard grid panels and exporting standalone HTML packages."""

    @staticmethod
    def export_html_dashboard(title: str, cards: List[Dict[str, str]], output_path: str) -> None:
        """Exports standalone responsive HTML dashboard package.

        Args:
            title: Dashboard title string.
            cards: List of dictionary cards containing label and value strings.
            output_path: Target HTML file path.
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cards_html = []
        for card in cards:
            cards_html.append(f"""
            <div style="background: #ffffff; border: 1px solid #e1e8ed; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); min-width: 200px; flex: 1;">
                <p style="color: #7f8c8d; font-size: 0.85em; font-weight: bold; margin: 0; text-transform: uppercase;">{card.get('label', '')}</p>
                <h2 style="color: #2c3e50; margin: 8px 0 0 0; font-size: 1.6em;">{card.get('value', '')}</h2>
            </div>
            """)

        html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>{title}</title>
    <style>
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background-color: #f8f9fa; margin: 0; padding: 30px; }}
        .header {{ margin-bottom: 25px; border-bottom: 2px solid #3498db; padding-bottom: 10px; }}
        .grid {{ display: flex; flex-wrap: wrap; gap: 20px; margin-bottom: 30px; }}
    </style>
</head>
<body>
    <div class="header">
        <h1 style="color: #2c3e50; margin: 0;">{title}</h1>
        <p style="color: #7f8c8d; margin-top: 5px;">Toxic Comment Classification, Sentiment Analysis & Emotion Mining System</p>
    </div>
    <div class="grid">
        {''.join(cards_html)}
    </div>
</body>
</html>"""

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html_content)
        logger.info(f"Exported interactive HTML dashboard to {output_path}")

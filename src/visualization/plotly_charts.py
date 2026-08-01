"""
Plotly Visualizer Module.

Builds interactive Plotly bar, line, pie, heatmap, radar, and scatter figures with hover tooltips and responsive layouts.
"""

import logging
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np

try:
    import plotly.graph_objects as go
    import plotly.express as px
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class PlotlyVisualizer:
    """Builder class constructing interactive Plotly figures."""

    @staticmethod
    def create_bar_chart(df: pd.DataFrame, x_col: str, y_col: str, title: str) -> Any:
        """Creates an interactive Plotly bar chart figure.

        Args:
            df: DataFrame containing plot data.
            x_col: X-axis column name.
            y_col: Y-axis column name.
            title: Chart title string.

        Returns:
            Plotly Figure object or dictionary mock.
        """
        if HAS_PLOTLY:
            fig = px.bar(df, x=x_col, y=y_col, title=title, template="plotly_white")
            fig.update_layout(title_x=0.5, font_family="Arial")
            return fig
        else:
            logger.warning("Plotly not installed; returning dictionary representation.")
            return {"type": "bar", "title": title, "data": df.to_dict(orient="records")}

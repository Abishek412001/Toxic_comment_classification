"""
Enterprise Analytics & Visualization Package (Phase 10).

Provides production-grade Plotly, Matplotlib, and Seaborn dashboards for Toxicity, Sentiment, Emotion, Performance, and XAI.
"""

from src.visualization.exceptions import (
    VisualizationError,
    ChartError,
    ThemeError,
    ExportError,
)
from src.visualization.config import VisualizationConfig
from src.visualization.theme import ThemeManager
from src.visualization.plotly_charts import PlotlyVisualizer
from src.visualization.matplotlib_charts import MatplotlibVisualizer
from src.visualization.seaborn_charts import SeabornVisualizer
from src.visualization.kpi_dashboard import KPIManager
from src.visualization.dashboard_manager import DashboardManager

__all__ = [
    "VisualizationError",
    "ChartError",
    "ThemeError",
    "ExportError",
    "VisualizationConfig",
    "ThemeManager",
    "PlotlyVisualizer",
    "MatplotlibVisualizer",
    "SeabornVisualizer",
    "KPIManager",
    "DashboardManager",
]

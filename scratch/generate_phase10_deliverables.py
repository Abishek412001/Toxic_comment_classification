"""
Master script to generate Phase 10 notebooks (84-92), visualization figures, and visualization summary report.
"""

import os
import sys
sys.path.insert(0, os.path.abspath("."))
import json
import logging

from src.visualization.toxicity_dashboard import ToxicityDashboard
from src.visualization.sentiment_dashboard import SentimentDashboard
from src.visualization.emotion_dashboard import EmotionDashboard
from src.visualization.model_performance_dashboard import ModelPerformanceDashboard
from src.visualization.xai_dashboard import XAIDashboardModule
from src.visualization.executive_kpi_dashboard import ExecutiveKPIDashboard
from src.visualization.plotly_dashboard_package import PlotlyDashboardPackage
from src.visualization.visualization_report_generator import VisualizationReportGenerator
from src.reports.visualization_summary import generate_visualization_master_dashboard, export_visualization_summary_report

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

NOTEBOOKS_DIR = "notebooks"
FIGURES_DIR = "outputs/figures"
REPORTS_DIR = "outputs/reports"

os.makedirs(NOTEBOOKS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

# 1. Generate Notebooks 84 through 92
notebook_configs = [
    ("84_toxicity_dashboard.ipynb", "Phase 10 - Step 112: Toxicity Analytics Dashboard", "from src.visualization.toxicity_dashboard import ToxicityDashboard\n\nToxicityDashboard.render_toxicity_dashboard()\nprint('Toxicity Dashboard Rendered!')"),
    ("85_sentiment_dashboard.ipynb", "Phase 10 - Step 113: Sentiment Analytics Dashboard", "from src.visualization.sentiment_dashboard import SentimentDashboard\n\nSentimentDashboard.render_sentiment_dashboard()\nprint('Sentiment Dashboard Rendered!')"),
    ("86_emotion_dashboard.ipynb", "Phase 10 - Step 114: Emotion Analytics Dashboard", "from src.visualization.emotion_dashboard import EmotionDashboard\n\nEmotionDashboard.render_emotion_dashboard()\nprint('Emotion Dashboard Rendered!')"),
    ("87_model_performance_dashboard.ipynb", "Phase 10 - Step 115: Model Performance & Leaderboard Dashboard", "from src.visualization.model_performance_dashboard import ModelPerformanceDashboard\n\nModelPerformanceDashboard.render_performance_dashboard()\nprint('Model Performance Dashboard Rendered!')"),
    ("88_xai_dashboard.ipynb", "Phase 10 - Step 116: Explainable AI Analytics Dashboard", "from src.visualization.xai_dashboard import XAIDashboardModule\n\nXAIDashboardModule.render_xai_dashboard()\nprint('XAI Dashboard Rendered!')"),
    ("89_executive_kpi_dashboard.ipynb", "Phase 10 - Step 117: Executive KPI & System Health Dashboard", "from src.visualization.executive_kpi_dashboard import ExecutiveKPIDashboard\n\nExecutiveKPIDashboard.render_kpi_dashboard()\nprint('Executive KPI Dashboard Rendered!')"),
    ("90_plotly_dashboards.ipynb", "Phase 10 - Step 118: Enterprise Interactive Plotly Dashboards Package", "from src.visualization.plotly_dashboard_package import PlotlyDashboardPackage\n\nPlotlyDashboardPackage.build_interactive_suite()\nprint('Interactive Plotly Suite Generated!')"),
    ("91_visualization_reports.ipynb", "Phase 10 - Step 119: Enterprise Dashboard Reports & Exports", "from src.visualization.visualization_report_generator import VisualizationReportGenerator\n\nVisualizationReportGenerator.export_markdown({'total_dashboards': 6}, 'outputs/reports/vis_test.md')\nprint('Visualization Reports Exported!')"),
    ("92_visualization_summary.ipynb", "Phase 10 - Step 120: Enterprise Visualization Final Report", "from src.reports.visualization_summary import generate_visualization_master_dashboard, export_visualization_summary_report\n\ngenerate_visualization_master_dashboard()\nexport_visualization_summary_report()\nprint('Master Visualization Final Report Generated Successfully!')"),
]

for filename, title, code in notebook_configs:
    filepath = os.path.join(NOTEBOOKS_DIR, filename)
    nb_json = {
        "cells": [
            {"cell_type": "markdown", "metadata": {}, "source": [f"# {title}\n"]},
            {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": ["import sys\nsys.path.append('..')\n\n" + code]}
        ],
        "metadata": {"language_info": {"name": "python"}},
        "nbformat": 4,
        "nbformat_minor": 2
    }
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(nb_json, f, indent=1)

logger.info(f"Generated {len(notebook_configs)} Phase 10 Jupyter Notebooks.")

# 2. Run Step 120 Executive Dashboards & Reports
ToxicityDashboard.render_toxicity_dashboard(os.path.join(FIGURES_DIR, "toxicity_analytics_dashboard.png"), os.path.join(REPORTS_DIR, "toxicity_analytics.html"))
SentimentDashboard.render_sentiment_dashboard(os.path.join(FIGURES_DIR, "sentiment_analytics_dashboard.png"), os.path.join(REPORTS_DIR, "sentiment_analytics.html"))
EmotionDashboard.render_emotion_dashboard(os.path.join(FIGURES_DIR, "emotion_analytics_dashboard.png"), os.path.join(REPORTS_DIR, "emotion_analytics.html"))
ModelPerformanceDashboard.render_performance_dashboard(os.path.join(FIGURES_DIR, "model_performance_dashboard.png"), os.path.join(REPORTS_DIR, "model_performance.html"))
XAIDashboardModule.render_xai_dashboard(os.path.join(FIGURES_DIR, "xai_analytics_dashboard.png"), os.path.join(REPORTS_DIR, "xai_analytics.html"))
ExecutiveKPIDashboard.render_kpi_dashboard(os.path.join(FIGURES_DIR, "executive_kpi_dashboard.png"), os.path.join(REPORTS_DIR, "executive_kpis.html"))
PlotlyDashboardPackage.build_interactive_suite(os.path.join(REPORTS_DIR, "interactive_plotly_suite.html"))

generate_visualization_master_dashboard(os.path.join(FIGURES_DIR, "visualization_master_dashboard.png"))
export_visualization_summary_report(
    os.path.join(REPORTS_DIR, "visualization_summary.md"),
    os.path.join(REPORTS_DIR, "visualization_summary.pdf")
)

logger.info("PHASE 10 DELIVERABLES GENERATED SUCCESSFULLY!")

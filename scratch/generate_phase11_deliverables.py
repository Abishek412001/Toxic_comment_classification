"""
Master script to generate Phase 11 notebook 93, Streamlit figures, and summary report.
"""

import os
import sys
sys.path.insert(0, os.path.abspath("."))
import json
import logging

from src.reports.streamlit_application_summary import generate_streamlit_master_dashboard, export_streamlit_application_summary_report

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

NOTEBOOKS_DIR = "notebooks"
FIGURES_DIR = "outputs/figures"
REPORTS_DIR = "outputs/reports"

os.makedirs(NOTEBOOKS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

# 1. Generate Notebook 93
notebook_configs = [
    ("93_streamlit_summary.ipynb", "Phase 11 - Step 130: Enterprise Streamlit Application Final Report", "from src.reports.streamlit_application_summary import generate_streamlit_master_dashboard, export_streamlit_application_summary_report\n\ngenerate_streamlit_master_dashboard()\nexport_streamlit_application_summary_report()\nprint('Master Streamlit Application Report Generated Successfully!')"),
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

logger.info(f"Generated Phase 11 Jupyter Notebook 93.")

# 2. Run Step 130 Executive Dashboard & Reports
generate_streamlit_master_dashboard(os.path.join(FIGURES_DIR, "streamlit_master_dashboard.png"))
export_streamlit_application_summary_report(
    os.path.join(REPORTS_DIR, "streamlit_application_summary.md"),
    os.path.join(REPORTS_DIR, "streamlit_application_summary.pdf")
)

logger.info("PHASE 11 DELIVERABLES GENERATED SUCCESSFULLY!")

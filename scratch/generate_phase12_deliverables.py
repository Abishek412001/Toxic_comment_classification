"""
Master script to generate Phase 12 notebook 94, MLOps figures, and summary report.
"""

import os
import sys
sys.path.insert(0, os.path.abspath("."))
import json
import logging

from src.reports.mlops_summary import generate_mlops_master_dashboard, export_mlops_summary_report

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

NOTEBOOKS_DIR = "notebooks"
FIGURES_DIR = "outputs/figures"
REPORTS_DIR = "outputs/reports"

os.makedirs(NOTEBOOKS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

# 1. Generate Notebook 94
notebook_configs = [
    ("94_mlops_summary.ipynb", "Phase 12 - Step 140: Enterprise MLOps Final Report", "from src.reports.mlops_summary import generate_mlops_master_dashboard, export_mlops_summary_report\n\ngenerate_mlops_master_dashboard()\nexport_mlops_summary_report()\nprint('Master MLOps & Deployment Report Generated Successfully!')"),
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

logger.info(f"Generated Phase 12 Jupyter Notebook 94.")

# 2. Run Step 140 Executive Dashboard & Reports
generate_mlops_master_dashboard(os.path.join(FIGURES_DIR, "mlops_master_dashboard.png"))
export_mlops_summary_report(
    os.path.join(REPORTS_DIR, "mlops_summary.md"),
    os.path.join(REPORTS_DIR, "mlops_summary.pdf")
)

logger.info("PHASE 12 DELIVERABLES GENERATED SUCCESSFULLY!")

"""
Master script to generate Phase 14 notebook 96, Documentation figures, and summary report.
"""

import os
import sys
sys.path.insert(0, os.path.abspath("."))
import json
import logging

from src.reports.documentation_summary import generate_documentation_master_dashboard, export_documentation_summary_report

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

NOTEBOOKS_DIR = "notebooks"
FIGURES_DIR = "outputs/figures"
DOCS_DIR = "docs"

os.makedirs(NOTEBOOKS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(DOCS_DIR, exist_ok=True)

# 1. Generate Notebook 96
notebook_configs = [
    ("96_documentation_summary.ipynb", "Phase 14 - Step 157: Enterprise Documentation Final Report", "from src.reports.documentation_summary import generate_documentation_master_dashboard, export_documentation_summary_report\n\ngenerate_documentation_master_dashboard()\nexport_documentation_summary_report()\nprint('Master Documentation & Portfolio Report Generated Successfully!')"),
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

logger.info(f"Generated Phase 14 Jupyter Notebook 96.")

# 2. Run Step 157 Executive Dashboard & Reports
generate_documentation_master_dashboard(os.path.join(FIGURES_DIR, "documentation_master_dashboard.png"))
export_documentation_summary_report(
    os.path.join(DOCS_DIR, "documentation_summary.md"),
    os.path.join(DOCS_DIR, "documentation_summary.pdf")
)

logger.info("PHASE 14 DELIVERABLES GENERATED SUCCESSFULLY!")

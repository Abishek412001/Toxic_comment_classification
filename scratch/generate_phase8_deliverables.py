"""
Master script to generate Phase 8 notebooks (66-74), emotion figures, and emotion mining summary report.
"""

import os
import sys
sys.path.insert(0, os.path.abspath("."))
import json
import logging
import pandas as pd

from src.emotion.nrc_analyzer import NRCEmotionAnalyzer
from src.emotion.transformer_analyzer import TransformerEmotionAnalyzer
from src.emotion.emotion_benchmark import EmotionBenchmarker
from src.emotion.emotion_dashboards import EmotionDashboard
from src.emotion.emotion_evaluator import EmotionEvaluator
from src.emotion.emotion_analytics_dashboard import EmotionAnalyticsDashboard
from src.reports.emotion_mining_summary import generate_emotion_master_dashboard, export_emotion_mining_summary_report

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

NOTEBOOKS_DIR = "notebooks"
FIGURES_DIR = "outputs/figures"
REPORTS_DIR = "outputs/reports"

os.makedirs(NOTEBOOKS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

# 1. Generate Notebooks 66 through 74
notebook_configs = [
    ("66_nrc_emotion.ipynb", "Phase 8 - Step 92: Production-Grade NRC Emotion Lexicon Analysis", "from src.emotion.nrc_analyzer import NRCEmotionAnalyzer\n\nanalyzer = NRCEmotionAnalyzer()\nres = analyzer.analyze('I feel mad and furious!')\nprint('NRC Result:', res)"),
    ("67_transformer_emotion.ipynb", "Phase 8 - Step 93: Production-Grade Transformer Emotion Detection", "from src.emotion.transformer_analyzer import TransformerEmotionAnalyzer\n\nanalyzer = TransformerEmotionAnalyzer()\nres = analyzer.analyze('I am super happy and thrilled!')\nprint('Transformer Result:', res)"),
    ("68_emotion_benchmark.ipynb", "Phase 8 - Step 94: Enterprise Emotion Benchmark Comparison", "from src.emotion.emotion_benchmark import EmotionBenchmarker\n\ndf = EmotionBenchmarker.benchmark_engines(['happy text', 'furious text'])\nEmotionBenchmarker.plot_benchmark_dashboard(df)\nprint('Emotion Benchmark Completed!')"),
    ("69_emotion_dashboards.ipynb", "Phase 8 - Step 95: Publication-Quality Emotion Dashboards", "from src.emotion.emotion_dashboards import EmotionDashboard\n\nresults = [{'emotion_label': 'joy', 'confidence_score': 0.85, 'probabilities': {'joy': 0.85, 'anger': 0.02, 'fear': 0.02, 'sadness': 0.02, 'surprise': 0.04, 'disgust': 0.01, 'neutral': 0.04}}]\nEmotionDashboard.render_distribution_dashboard(results)\nprint('Emotion Dashboard Rendered!')"),
    ("70_emotion_evaluation.ipynb", "Phase 8 - Step 96: Multi-Class Emotion Model Evaluation & ROC Curves", "from src.emotion.emotion_evaluator import EmotionEvaluator\n\neval_data = EmotionEvaluator.evaluate_predictions(['joy', 'anger'], ['joy', 'anger'])\nEmotionEvaluator.plot_evaluation_charts(eval_data)\nprint('Emotion Evaluation Completed!')"),
    ("71_production_emotion_pipeline.ipynb", "Phase 8 - Step 97: Enterprise Production Emotion Pipeline", "from src.emotion.production_emotion_pipeline import ProductionEmotionPipeline\n\npipeline = ProductionEmotionPipeline()\nres = pipeline.predict_single('Awesome work!')\nprint('Production Payload:', pipeline.format_rest_payload(res))"),
    ("72_emotion_reports.ipynb", "Phase 8 - Step 98: Enterprise Emotion Reports & Exports", "from src.emotion.emotion_report_generator import EmotionReportGenerator\n\nEmotionReportGenerator.export_markdown({'joy_count': 150}, 'outputs/reports/emotion_test.md')\nprint('Emotion Reports Exported!')"),
    ("73_emotion_analytics_dashboard.ipynb", "Phase 8 - Step 99: Enterprise Emotion Analytics Dashboard", "from src.emotion.emotion_analytics_dashboard import EmotionAnalyticsDashboard\nimport pandas as pd\n\ndf = pd.DataFrame([{'emotion_label': 'anger', 'confidence_score': 0.9}])\nEmotionAnalyticsDashboard.render_analytics_dashboard(df)\nprint('Emotion Analytics Dashboard Rendered!')"),
    ("74_emotion_summary.ipynb", "Phase 8 - Step 100: Enterprise Emotion Mining Final Report", "from src.reports.emotion_mining_summary import generate_emotion_master_dashboard, export_emotion_mining_summary_report\n\ngenerate_emotion_master_dashboard()\nexport_emotion_mining_summary_report()\nprint('Master Emotion Final Report Generated Successfully!')"),
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

logger.info(f"Generated {len(notebook_configs)} Phase 8 Jupyter Notebooks.")

# 2. Run Step 100 Executive Dashboard & Reports
bench_df = EmotionBenchmarker.benchmark_engines(["great comment", "furious comment"])
EmotionBenchmarker.plot_benchmark_dashboard(bench_df, os.path.join(FIGURES_DIR, "emotion_benchmark_dashboard.png"))

res_sample = [
    {"emotion_label": "joy", "confidence_score": 0.85, "probabilities": {"joy": 0.85, "anger": 0.02, "fear": 0.02, "sadness": 0.02, "surprise": 0.04, "disgust": 0.01, "neutral": 0.04}},
    {"emotion_label": "anger", "confidence_score": 0.88, "probabilities": {"joy": 0.01, "anger": 0.88, "fear": 0.03, "sadness": 0.02, "surprise": 0.01, "disgust": 0.04, "neutral": 0.01}},
]
EmotionDashboard.render_distribution_dashboard(res_sample, os.path.join(FIGURES_DIR, "emotion_distribution_dashboard.png"))

eval_sample = EmotionEvaluator.evaluate_predictions(["joy", "anger", "neutral"], ["joy", "anger", "neutral"])
EmotionEvaluator.plot_evaluation_charts(eval_sample, os.path.join(FIGURES_DIR, "emotion_evaluation_roc.png"))

analytics_df = pd.DataFrame(res_sample)
EmotionAnalyticsDashboard.render_analytics_dashboard(analytics_df, os.path.join(FIGURES_DIR, "emotion_analytics_dashboard.png"))

generate_emotion_master_dashboard(os.path.join(FIGURES_DIR, "emotion_master_dashboard.png"))
export_emotion_mining_summary_report(
    os.path.join(REPORTS_DIR, "emotion_mining_summary.md"),
    os.path.join(REPORTS_DIR, "emotion_mining_summary.pdf")
)

logger.info("PHASE 8 DELIVERABLES GENERATED SUCCESSFULLY!")

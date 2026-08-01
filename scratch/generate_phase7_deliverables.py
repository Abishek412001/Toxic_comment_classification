"""
Master script to generate Phase 7 notebooks (57-65), sentiment figures, and sentiment summary report.
"""

import os
import sys
sys.path.insert(0, os.path.abspath("."))
import json
import logging

from src.sentiment.vader_analyzer import VADERAnalyzer
from src.sentiment.textblob_analyzer import TextBlobAnalyzer
from src.sentiment.transformer_analyzer import TransformerSentimentAnalyzer
from src.sentiment.sentiment_benchmark import SentimentBenchmarker
from src.sentiment.sentiment_dashboards import SentimentDashboard
from src.sentiment.sentiment_evaluator import SentimentEvaluator
from src.reports.sentiment_analysis_summary import generate_sentiment_master_dashboard, export_sentiment_analysis_summary_report

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

NOTEBOOKS_DIR = "notebooks"
FIGURES_DIR = "outputs/figures"
REPORTS_DIR = "outputs/reports"

os.makedirs(NOTEBOOKS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

# 1. Generate Notebooks 57 through 65
notebook_configs = [
    ("57_vader_sentiment.ipynb", "Phase 7 - Step 82: Production-Grade VADER Sentiment Analysis", "from src.sentiment.vader_analyzer import VADERAnalyzer\n\nanalyzer = VADERAnalyzer()\nres = analyzer.analyze('This is an awesome and positive comment!')\nprint('VADER Result:', res)"),
    ("58_textblob_sentiment.ipynb", "Phase 7 - Step 83: Production-Grade TextBlob Sentiment Analysis", "from src.sentiment.textblob_analyzer import TextBlobAnalyzer\n\nanalyzer = TextBlobAnalyzer()\nres = analyzer.analyze('This is a terrible experience.')\nprint('TextBlob Result:', res)"),
    ("59_transformer_sentiment.ipynb", "Phase 7 - Step 84: Production-Grade Transformer Sentiment Analysis", "from src.sentiment.transformer_analyzer import TransformerSentimentAnalyzer\n\nanalyzer = TransformerSentimentAnalyzer()\nres = analyzer.analyze('Great product and high quality!')\nprint('Transformer Result:', res)"),
    ("60_sentiment_benchmark.ipynb", "Phase 7 - Step 85: Enterprise Sentiment Benchmark Comparison", "from src.sentiment.sentiment_benchmark import SentimentBenchmarker\n\ndf = SentimentBenchmarker.benchmark_engines(['good text', 'bad text'])\nSentimentBenchmarker.plot_benchmark_dashboard(df)\nprint('Sentiment Benchmark Completed!')"),
    ("61_sentiment_dashboards.ipynb", "Phase 7 - Step 86: Publication-Quality Sentiment Dashboards", "from src.sentiment.sentiment_dashboards import SentimentDashboard\n\nresults = [{'sentiment_label': 'positive', 'compound_score': 0.8, 'confidence_score': 0.8}]\nSentimentDashboard.render_distribution_dashboard(results)\nprint('Sentiment Dashboard Rendered!')"),
    ("62_sentiment_evaluation.ipynb", "Phase 7 - Step 87: Multi-Class Sentiment Evaluation & ROC Curves", "from src.sentiment.sentiment_evaluator import SentimentEvaluator\n\neval_data = SentimentEvaluator.evaluate_predictions(['positive', 'negative'], ['positive', 'negative'])\nSentimentEvaluator.plot_evaluation_charts(eval_data)\nprint('Sentiment Evaluation Completed!')"),
    ("63_production_sentiment_pipeline.ipynb", "Phase 7 - Step 88: Enterprise Production Sentiment Pipeline", "from src.sentiment.production_sentiment_pipeline import ProductionSentimentPipeline\n\npipeline = ProductionSentimentPipeline()\nres = pipeline.predict_single('Awesome pipeline!')\nprint('Production Payload:', pipeline.format_rest_payload(res))"),
    ("64_sentiment_reports.ipynb", "Phase 7 - Step 89: Enterprise Sentiment Reports & Exports", "from src.sentiment.sentiment_report_generator import SentimentReportGenerator\n\nSentimentReportGenerator.export_markdown({'positive_count': 100}, 'outputs/reports/sentiment_test.md')\nprint('Sentiment Reports Exported!')"),
    ("65_sentiment_summary.ipynb", "Phase 7 - Step 90: Enterprise Sentiment Analysis Final Report", "from src.reports.sentiment_analysis_summary import generate_sentiment_master_dashboard, export_sentiment_analysis_summary_report\n\ngenerate_sentiment_master_dashboard()\nexport_sentiment_analysis_summary_report()\nprint('Master Sentiment Final Report Generated Successfully!')"),
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

logger.info(f"Generated {len(notebook_configs)} Phase 7 Jupyter Notebooks.")

# 2. Run Step 90 Executive Dashboard & Reports
bench_df = SentimentBenchmarker.benchmark_engines(["great comment", "bad comment"])
SentimentBenchmarker.plot_benchmark_dashboard(bench_df, os.path.join(FIGURES_DIR, "sentiment_benchmark_dashboard.png"))

res_sample = [
    {"sentiment_label": "positive", "compound_score": 0.85, "confidence_score": 0.85},
    {"sentiment_label": "negative", "compound_score": -0.75, "confidence_score": 0.75},
    {"sentiment_label": "neutral", "compound_score": 0.00, "confidence_score": 0.60},
]
SentimentDashboard.render_distribution_dashboard(res_sample, os.path.join(FIGURES_DIR, "sentiment_distribution_dashboard.png"))

eval_sample = SentimentEvaluator.evaluate_predictions(["positive", "negative", "neutral"], ["positive", "negative", "neutral"])
SentimentEvaluator.plot_evaluation_charts(eval_sample, os.path.join(FIGURES_DIR, "sentiment_evaluation_roc.png"))

generate_sentiment_master_dashboard(os.path.join(FIGURES_DIR, "sentiment_master_dashboard.png"))
export_sentiment_analysis_summary_report(
    os.path.join(REPORTS_DIR, "sentiment_analysis_summary.md"),
    os.path.join(REPORTS_DIR, "sentiment_analysis_summary.pdf")
)

logger.info("PHASE 7 DELIVERABLES GENERATED SUCCESSFULLY!")

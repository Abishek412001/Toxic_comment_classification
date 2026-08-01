"""
Master script to generate Phase 5 notebooks (39-55), evaluation figures, and model development summary report.
"""

import os
import sys
sys.path.insert(0, os.path.abspath("."))
import json
import logging

from src.reports.traditional_ml_summary import export_traditional_ml_report
from src.reports.deep_learning_summary import export_deep_learning_report
from src.reports.transformer_summary import export_transformer_report
from src.reports.model_development_summary import generate_master_model_dashboard, export_model_development_summary_report

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

NOTEBOOKS_DIR = "notebooks"
FIGURES_DIR = "outputs/figures"
REPORTS_DIR = "outputs/reports"

os.makedirs(NOTEBOOKS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

# 1. Generate Notebooks 39 through 55
notebook_configs = [
    ("39_data_splitting.ipynb", "Phase 5 - Step 53: Multi-Label Data Splitting Strategy", "from src.models.data_splitter import DataSplitter\nimport numpy as np\n\nsplitter = DataSplitter()\nX = np.random.randn(100, 10)\ny = np.random.randint(0, 2, size=(100, 6))\nX_tr, X_va, X_te, y_tr, y_va, y_te = splitter.split(X, y)\nprint('Train:', len(y_tr), 'Val:', len(y_va), 'Test:', len(y_te))"),
    ("40_dummy_classifier.ipynb", "Phase 5 - Step 54: Dummy Baseline Classifier", "from src.models.dummy_classifier import DummyBaselineClassifier\nimport numpy as np\n\nmodel = DummyBaselineClassifier()\nX = np.random.randn(10, 5)\ny = np.random.randint(0, 2, size=(10, 6))\nmodel.fit(X, y)\nprint('Dummy Prediction Shape:', model.predict(X).shape)"),
    ("41_logistic_regression.ipynb", "Phase 5 - Step 55: Multi-Label Logistic Regression", "from src.models.logistic_regression import MultiLabelLogisticRegression\nimport numpy as np\n\nmodel = MultiLabelLogisticRegression()\nX = np.random.randn(10, 5)\ny = np.random.randint(0, 2, size=(10, 6))\nmodel.fit(X, y)\nprint('LR Prediction Shape:', model.predict(X).shape)"),
    ("42_naive_bayes.ipynb", "Phase 5 - Step 56: Multinomial Naive Bayes", "from src.models.naive_bayes import MultiLabelNaiveBayes\nimport numpy as np\n\nmodel = MultiLabelNaiveBayes()\nX = np.abs(np.random.randn(10, 5))\ny = np.random.randint(0, 2, size=(10, 6))\nmodel.fit(X, y)\nprint('NB Prediction Shape:', model.predict(X).shape)"),
    ("43_random_forest.ipynb", "Phase 5 - Step 57: Multi-Label Random Forest", "from src.models.random_forest import MultiLabelRandomForest\nimport numpy as np\n\nmodel = MultiLabelRandomForest()\nX = np.random.randn(10, 5)\ny = np.random.randint(0, 2, size=(10, 6))\nmodel.fit(X, y)\nprint('RF Prediction Shape:', model.predict(X).shape)"),
    ("44_xgboost.ipynb", "Phase 5 - Step 58: Multi-Label XGBoost", "from src.models.xgboost_model import MultiLabelXGBoost\nimport numpy as np\n\nmodel = MultiLabelXGBoost()\nX = np.random.randn(10, 5)\ny = np.random.randint(0, 2, size=(10, 6))\nmodel.fit(X, y)\nprint('XGB Prediction Shape:', model.predict(X).shape)"),
    ("45_lightgbm.ipynb", "Phase 5 - Step 59: Multi-Label LightGBM", "from src.models.lightgbm_model import MultiLabelLightGBM\nimport numpy as np\n\nmodel = MultiLabelLightGBM()\nX = np.random.randn(10, 5)\ny = np.random.randint(0, 2, size=(10, 6))\nmodel.fit(X, y)\nprint('LGB Prediction Shape:', model.predict(X).shape)"),
    ("46_traditional_ml_benchmark.ipynb", "Phase 5 - Step 60: Traditional ML Benchmark Suite", "from src.reports.traditional_ml_summary import export_traditional_ml_report\n\nexport_traditional_ml_report()\nprint('Traditional ML Benchmark Exported!')"),
    ("47_lstm.ipynb", "Phase 5 - Step 61: Multi-Label LSTM Model", "from src.models.lstm_model import MultiLabelLSTM\nimport numpy as np\n\nmodel = MultiLabelLSTM()\nX = np.random.randn(10, 5)\ny = np.random.randint(0, 2, size=(10, 6))\nmodel.fit(X, y)\nprint('LSTM Prediction Shape:', model.predict(X).shape)"),
    ("48_bilstm.ipynb", "Phase 5 - Step 62: Multi-Label Bidirectional LSTM", "from src.models.bilstm_model import MultiLabelBiLSTM\nimport numpy as np\n\nmodel = MultiLabelBiLSTM()\nX = np.random.randn(10, 5)\ny = np.random.randint(0, 2, size=(10, 6))\nmodel.fit(X, y)\nprint('BiLSTM Prediction Shape:', model.predict(X).shape)"),
    ("49_gru.ipynb", "Phase 5 - Step 63: Multi-Label GRU Model", "from src.models.gru_model import MultiLabelGRU\nimport numpy as np\n\nmodel = MultiLabelGRU()\nX = np.random.randn(10, 5)\ny = np.random.randint(0, 2, size=(10, 6))\nmodel.fit(X, y)\nprint('GRU Prediction Shape:', model.predict(X).shape)"),
    ("50_deep_learning_benchmark.ipynb", "Phase 5 - Step 64: Deep Learning Benchmark Suite", "from src.reports.deep_learning_summary import export_deep_learning_report\n\nexport_deep_learning_report()\nprint('Deep Learning Benchmark Exported!')"),
    ("51_bert_finetuning.ipynb", "Phase 5 - Step 65: Multi-Label BERT Fine-Tuning", "from src.models.bert_classifier import MultiLabelBERTClassifier\nimport numpy as np\n\nmodel = MultiLabelBERTClassifier()\nX = np.random.randn(10, 5)\ny = np.random.randint(0, 2, size=(10, 6))\nmodel.fit(X, y)\nprint('BERT Prediction Shape:', model.predict(X).shape)"),
    ("52_distilbert.ipynb", "Phase 5 - Step 66: Multi-Label DistilBERT Fine-Tuning", "from src.models.distilbert_classifier import MultiLabelDistilBERTClassifier\nimport numpy as np\n\nmodel = MultiLabelDistilBERTClassifier()\nX = np.random.randn(10, 5)\ny = np.random.randint(0, 2, size=(10, 6))\nmodel.fit(X, y)\nprint('DistilBERT Prediction Shape:', model.predict(X).shape)"),
    ("53_roberta.ipynb", "Phase 5 - Step 67: Multi-Label RoBERTa Fine-Tuning", "from src.models.roberta_classifier import MultiLabelRoBERTaClassifier\nimport numpy as np\n\nmodel = MultiLabelRoBERTaClassifier()\nX = np.random.randn(10, 5)\ny = np.random.randint(0, 2, size=(10, 6))\nmodel.fit(X, y)\nprint('RoBERTa Prediction Shape:', model.predict(X).shape)"),
    ("54_transformer_benchmark.ipynb", "Phase 5 - Step 68: Transformer Benchmark Suite", "from src.reports.transformer_summary import export_transformer_report\n\nexport_transformer_report()\nprint('Transformer Benchmark Exported!')"),
    ("55_model_development_summary.ipynb", "Phase 5 - Step 71: Enterprise Model Development Summary", "from src.reports.model_development_summary import generate_master_model_dashboard, export_model_development_summary_report\n\ngenerate_master_model_dashboard()\nexport_model_development_summary_report()\nprint('Master Model Development Summary Generated Successfully!')"),
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

logger.info(f"Generated {len(notebook_configs)} Phase 5 Jupyter Notebooks.")

# 2. Run Step 71 Executive Dashboard & Reports
export_traditional_ml_report()
export_deep_learning_report()
export_transformer_report()
generate_master_model_dashboard(os.path.join(FIGURES_DIR, "model_comparison_dashboard.png"))
export_model_development_summary_report(
    os.path.join(REPORTS_DIR, "model_development_summary.md"),
    os.path.join(REPORTS_DIR, "model_development_summary.pdf")
)

logger.info("PHASE 5 DELIVERABLES GENERATED SUCCESSFULLY!")

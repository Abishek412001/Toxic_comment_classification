"""
Master script to generate Phase 4 notebooks (30-38), evaluation figures, and feature summary report.
"""

import os
import sys
sys.path.insert(0, os.path.abspath("."))
import json
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from src.utils.data_loader import load_toxic_comment_data
from src.preprocessing.pipeline import build_pipeline
from src.features.bow_vectorizer import BoWVectorizer
from src.features.tfidf_vectorizer import TFIDFFeatureExtractor
from src.features.word2vec import Word2VecFeatureExtractor
from src.features.fasttext import FastTextFeatureExtractor
from src.features.glove import GloVeFeatureExtractor
from src.features.bert_embeddings import BERTEmbeddingExtractor
from src.features.sentence_transformer import SentenceTransformerExtractor
from src.features.feature_selection import FeatureSelector
from src.reports.feature_engineering_summary import generate_feature_engineering_dashboard, export_feature_summary_report

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

NOTEBOOKS_DIR = "notebooks"
FIGURES_DIR = "outputs/figures"
REPORTS_DIR = "outputs/reports"

os.makedirs(NOTEBOOKS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

# 1. Generate Notebooks 30 through 38
notebook_configs = [
    ("30_bag_of_words.ipynb", "Phase 4 - Step 42: Bag of Words (BoW) Vectorizer", "from src.features.bow_vectorizer import BoWVectorizer\n\nextractor = BoWVectorizer()\nmatrix = extractor.fit_transform(['clean toxic text', 'another comment'])\nprint('BoW Shape:', matrix.shape)"),
    ("31_tfidf.ipynb", "Phase 4 - Step 43: TF-IDF Feature Extraction", "from src.features.tfidf_vectorizer import TFIDFFeatureExtractor\n\nextractor = TFIDFFeatureExtractor()\nmatrix = extractor.fit_transform(['clean toxic text', 'another comment'])\nprint('TF-IDF Shape:', matrix.shape)"),
    ("32_word2vec.ipynb", "Phase 4 - Step 44: Word2Vec Dense Embeddings", "from src.features.word2vec import Word2VecFeatureExtractor\n\nextractor = Word2VecFeatureExtractor()\nmatrix = extractor.fit_transform(['clean toxic text', 'another comment'])\nprint('Word2Vec Shape:', matrix.shape)"),
    ("33_fasttext.ipynb", "Phase 4 - Step 45: FastText Subword Embeddings", "from src.features.fasttext import FastTextFeatureExtractor\n\nextractor = FastTextFeatureExtractor()\nmatrix = extractor.fit_transform(['clean toxic text', 'another comment'])\nprint('FastText Shape:', matrix.shape)"),
    ("34_glove.ipynb", "Phase 4 - Step 46: GloVe Co-occurrence Embeddings", "from src.features.glove import GloVeFeatureExtractor\n\nextractor = GloVeFeatureExtractor()\nmatrix = extractor.fit_transform(['clean toxic text', 'another comment'])\nprint('GloVe Shape:', matrix.shape)"),
    ("35_bert_embeddings.ipynb", "Phase 4 - Step 47: BERT Contextual Embeddings", "from src.features.bert_embeddings import BERTEmbeddingExtractor\n\nextractor = BERTEmbeddingExtractor()\nmatrix = extractor.fit_transform(['clean toxic text', 'another comment'])\nprint('BERT Shape:', matrix.shape)"),
    ("36_sentence_transformers.ipynb", "Phase 4 - Step 48: Sentence Transformer Embeddings", "from src.features.sentence_transformer import SentenceTransformerExtractor\n\nextractor = SentenceTransformerExtractor()\nmatrix = extractor.fit_transform(['clean toxic text', 'another comment'])\nprint('Sentence Transformer Shape:', matrix.shape)"),
    ("37_feature_selection.ipynb", "Phase 4 - Step 50: Feature Selection Engine", "from src.features.tfidf_vectorizer import TFIDFFeatureExtractor\nfrom src.features.feature_selection import FeatureSelector\nimport numpy as np\n\nextractor = TFIDFFeatureExtractor()\nmatrix = extractor.fit_transform(['clean toxic text', 'another comment'])\nselector = FeatureSelector(method='chi2', k=5)\nX_sel = selector.fit_transform(matrix, np.array([1, 0]))\nprint('Selected Shape:', X_sel.shape)"),
    ("38_feature_engineering_summary.ipynb", "Phase 4 - Step 51: Enterprise Feature Engineering Summary", "from src.reports.feature_engineering_summary import generate_feature_engineering_dashboard, export_feature_summary_report\n\ngenerate_feature_engineering_dashboard()\nexport_feature_summary_report()\nprint('Master Feature Engineering Summary Generated Successfully!')"),
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

logger.info(f"Generated {len(notebook_configs)} Phase 4 Jupyter Notebooks.")

# 2. Run Step 51 Executive Dashboard & Reports
generate_feature_engineering_dashboard(os.path.join(FIGURES_DIR, "feature_engineering_dashboard.png"))
export_feature_summary_report(
    os.path.join(REPORTS_DIR, "feature_engineering_summary.md"),
    os.path.join(REPORTS_DIR, "feature_engineering_summary.pdf")
)

logger.info("PHASE 4 DELIVERABLES GENERATED SUCCESSFULLY!")

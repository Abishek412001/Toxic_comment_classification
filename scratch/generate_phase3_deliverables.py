"""
Master script to generate Phase 3 notebooks (16-29), evaluation figures, and preprocessing report.
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
from src.preprocessing.evaluator import PipelineEvaluator
from src.preprocessing.benchmark import PipelineBenchmark

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

NOTEBOOKS_DIR = "notebooks"
FIGURES_DIR = "outputs/figures"
REPORTS_DIR = "outputs/reports"

os.makedirs(NOTEBOOKS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

# 1. Generate Notebooks 16 through 29
notebook_configs = [
    ("16_lowercase.ipynb", "Phase 3 - Step 27: Lowercasing Module", "from src.preprocessing.lowercase import apply_lowercase, batch_lowercase\n\nraw = 'DON\\'T Shout HERE! Café 123'\nclean = apply_lowercase(raw)\nprint(f'Before: {raw}\\nAfter: {clean}')"),
    ("17_contraction_expansion.ipynb", "Phase 3 - Step 28: Contraction Expansion Module", "from src.preprocessing.contractions import expand_contractions\n\nraw = 'I don\\'t know why he can\\'t come'\nclean = expand_contractions(raw)\nprint(f'Before: {raw}\\nAfter: {clean}')"),
    ("18_html_removal.ipynb", "Phase 3 - Step 29: HTML Tag Removal Module", "from src.preprocessing.html_cleaner import remove_html_tags\n\nraw = '<p>This is <b>bold</b> text &amp; tags</p>'\nclean = remove_html_tags(raw)\nprint(f'Before: {raw}\\nAfter: {clean}')"),
    ("19_url_removal.ipynb", "Phase 3 - Step 30: URL Removal Module", "from src.preprocessing.url_cleaner import remove_urls\n\nraw = 'Check out http://example.com for info'\nclean = remove_urls(raw, replacement_token='[URL]')\nprint(f'Before: {raw}\\nAfter: {clean}')"),
    ("20_email_removal.ipynb", "Phase 3 - Step 31: Email Address Removal Module", "from src.preprocessing.email_cleaner import remove_emails\n\nraw = 'Contact support@domain.com now'\nclean = remove_emails(raw, replacement_token='[EMAIL]')\nprint(f'Before: {raw}\\nAfter: {clean}')"),
    ("21_emoji_removal.ipynb", "Phase 3 - Step 32: Emoji Removal / Conversion Module", "from src.preprocessing.emoji_cleaner import remove_emojis\n\nraw = 'Angry text 🤬 🔥'\nclean = remove_emojis(raw, demoji_to_text=True)\nprint(f'Before: {raw}\\nAfter: {clean}')"),
    ("22_number_removal.ipynb", "Phase 3 - Step 33: Number Removal Module", "from src.preprocessing.number_cleaner import remove_numbers\n\nraw = 'User 123 scored 98.5 percent'\nclean = remove_numbers(raw, replacement_token='0')\nprint(f'Before: {raw}\\nAfter: {clean}')"),
    ("23_punctuation_removal.ipynb", "Phase 3 - Step 34: Punctuation Removal Module", "from src.preprocessing.punctuation_cleaner import remove_punctuation\n\nraw = 'Hello, world!!! How are you?'\nclean = remove_punctuation(raw)\nprint(f'Before: {raw}\\nAfter: {clean}')"),
    ("24_special_character_removal.ipynb", "Phase 3 - Step 35: Special Character Removal Module", "from src.preprocessing.special_character_cleaner import remove_special_characters\n\nraw = 'Clean § © ™ symbol text'\nclean = remove_special_characters(raw)\nprint(f'Before: {raw}\\nAfter: {clean}')"),
    ("25_whitespace_normalization.ipynb", "Phase 3 - Step 36: Whitespace Normalization Module", "from src.preprocessing.whitespace_normalizer import normalize_whitespace\n\nraw = '  Too   many   spaces \\n\\t here  '\nclean = normalize_whitespace(raw)\nprint(f'Before: {raw}\\nAfter: {clean}')"),
    ("26_stopword_removal.ipynb", "Phase 3 - Step 37: Stopword Removal Module", "from src.preprocessing.stopword_remover import remove_stopwords\n\nraw = 'this is a toxic comment on wikipedia article'\nclean = remove_stopwords(raw)\nprint(f'Before: {raw}\\nAfter: {clean}')"),
    ("27_lemmatization.ipynb", "Phase 3 - Step 38: Lemmatization Module", "from src.preprocessing.lemmatizer import lemmatize_text\n\nraw = 'running cars and feet'\nclean = lemmatize_text(raw, backend='spacy')\nprint(f'Before: {raw}\\nAfter: {clean}')"),
    ("28_complete_pipeline.ipynb", "Phase 3 - Step 39: Complete Configurable Preprocessing Pipeline", "from src.preprocessing.pipeline import build_pipeline\n\npipeline = build_pipeline('traditional_ml')\nraw = '<p>DON\\'T click http://test.com! 🤬 123</p>'\nclean = pipeline.transform(raw)\nprint(f'Before: {raw}\\nAfter: {clean}')"),
    ("29_pipeline_evaluation.ipynb", "Phase 3 - Step 40: Pipeline Evaluation & Productionization", "from src.preprocessing.pipeline import build_pipeline\nfrom src.preprocessing.evaluator import PipelineEvaluator\nfrom src.utils.data_loader import load_toxic_comment_data\n\ndf = load_toxic_comment_data()\npipeline = build_pipeline('traditional_ml')\ncleaned = pipeline.transform_batch(df['comment_text'])\neval_metrics = PipelineEvaluator.evaluate_texts(df['comment_text'].tolist(), cleaned)\nprint('Evaluation Metrics:', eval_metrics)"),
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

logger.info(f"Generated {len(notebook_configs)} Phase 3 Jupyter Notebooks.")

# 2. Run Step 40 Benchmarking & Evaluation Pipeline
df = load_toxic_comment_data()
raw_texts = df["comment_text"].tolist()

pipeline = build_pipeline("traditional_ml")
cleaned_texts = pipeline.transform_batch(raw_texts, n_jobs=1)

eval_metrics = PipelineEvaluator.evaluate_texts(raw_texts, cleaned_texts)
bench_metrics = PipelineBenchmark.run_benchmark(pipeline, raw_texts, batch_sizes=[10, 100, 500])

# 3. Generate 4 Publication-Quality 300 DPI Visualizations

# Figure 1: Preprocessing Before vs After Comparison Bar Chart
plt.figure(figsize=(10, 6))
categories = ["Characters", "Total Tokens", "Unique Vocabulary"]
raw_vals = [eval_metrics["total_raw_characters"], eval_metrics["total_raw_tokens"], eval_metrics["raw_vocabulary_size"]]
clean_vals = [eval_metrics["total_clean_characters"], eval_metrics["total_clean_tokens"], eval_metrics["clean_vocabulary_size"]]

x = np.arange(len(categories))
width = 0.35

plt.bar(x - width/2, raw_vals, width, label="Raw Text", color="#e74c3c")
plt.bar(x + width/2, clean_vals, width, label="Cleaned Text", color="#2ecc71")

plt.title("Text Corpus Metrics: Before vs After Preprocessing", fontsize=14, fontweight="bold", pad=12)
plt.ylabel("Count Volume", fontsize=12)
plt.xticks(x, categories, fontsize=11)
plt.legend(fontsize=11)
plt.yscale("log")
plt.grid(axis="y", linestyle="--", alpha=0.7)

for i in range(len(categories)):
    plt.annotate(f"{raw_vals[i]:,}", (x[i] - width/2, raw_vals[i]), ha="center", va="bottom", fontsize=9)
    plt.annotate(f"{clean_vals[i]:,}", (x[i] + width/2, clean_vals[i]), ha="center", va="bottom", fontsize=9)

plt.tight_layout()
fig1_path = os.path.join(FIGURES_DIR, "preprocessing_comparison.png")
plt.savefig(fig1_path, dpi=300)
plt.close()
logger.info(f"Saved {fig1_path}")

# Figure 2: Execution Time & Throughput Benchmark
plt.figure(figsize=(9, 5))
batches = ["Batch 10", "Batch 100", "Batch 500"]
throughputs = [bench_metrics["batch_benchmarks"][f"batch_{b}"]["throughput_docs_per_sec"] for b in [10, 100, 500]]

sns.barplot(x=batches, y=throughputs, palette="crest")
plt.title("Pipeline Processing Throughput Across Batch Sizes", fontsize=14, fontweight="bold", pad=12)
plt.xlabel("Batch Size Configuration", fontsize=12)
plt.ylabel("Throughput (Comments / Second)", fontsize=12)
plt.grid(axis="y", linestyle="--", alpha=0.7)

for idx, val in enumerate(throughputs):
    plt.annotate(f"{val:,.1f} docs/sec", (idx, val), ha="center", va="bottom", fontsize=10, xytext=(0, 3), textcoords="offset points")

plt.tight_layout()
fig2_path = os.path.join(FIGURES_DIR, "execution_time.png")
plt.savefig(fig2_path, dpi=300)
plt.close()
logger.info(f"Saved {fig2_path}")

# Figure 3: Token Reduction Statistics
plt.figure(figsize=(7, 5))
tok_data = pd.DataFrame({"Stage": ["Raw Tokens", "Clean Tokens"], "Count": [eval_metrics["total_raw_tokens"], eval_metrics["total_clean_tokens"]]})
sns.barplot(x="Stage", y="Count", data=tok_data, palette=["#3498db", "#9b59b6"])
plt.title(f"Token Reduction Ratio ({eval_metrics['token_reduction_pct']}% Pruned)", fontsize=14, fontweight="bold", pad=12)
plt.ylabel("Total Token Count", fontsize=12)
plt.grid(axis="y", linestyle="--", alpha=0.7)

for p in plt.gca().patches:
    val = int(p.get_height())
    plt.gca().annotate(f"{val:,}", (p.get_x() + p.get_width() / 2., val), ha="center", va="bottom", fontsize=11, xytext=(0, 3), textcoords="offset points")

plt.tight_layout()
fig3_path = os.path.join(FIGURES_DIR, "token_reduction.png")
plt.savefig(fig3_path, dpi=300)
plt.close()
logger.info(f"Saved {fig3_path}")

# Figure 4: Vocabulary Compression Statistics
plt.figure(figsize=(7, 5))
voc_data = pd.DataFrame({"Stage": ["Raw Vocab", "Clean Vocab"], "Count": [eval_metrics["raw_vocabulary_size"], eval_metrics["clean_vocabulary_size"]]})
sns.barplot(x="Stage", y="Count", data=voc_data, palette=["#e67e22", "#1abc9c"])
plt.title(f"Vocabulary Compression Ratio ({eval_metrics['vocabulary_reduction_pct']}% Reduced)", fontsize=14, fontweight="bold", pad=12)
plt.ylabel("Unique Vocabulary Words", fontsize=12)
plt.grid(axis="y", linestyle="--", alpha=0.7)

for p in plt.gca().patches:
    val = int(p.get_height())
    plt.gca().annotate(f"{val:,}", (p.get_x() + p.get_width() / 2., val), ha="center", va="bottom", fontsize=11, xytext=(0, 3), textcoords="offset points")

plt.tight_layout()
fig4_path = os.path.join(FIGURES_DIR, "vocabulary_reduction.png")
plt.savefig(fig4_path, dpi=300)
plt.close()
logger.info(f"Saved {fig4_path}")

# 4. Generate Master Enterprise Preprocessing Markdown Report
report_md = f"""# Toxic Comment Classification System - Phase 3 Preprocessing Master Report

## 1. Executive Summary & Overview Metrics

- **Pipeline Name**: Enterprise Configurable Text Preprocessing Architecture
- **Supported Paradigms**: Traditional ML, Deep Learning (RNN/LSTM), Transformers (BERT/RoBERTa), Real-time Streamlit API
- **Total Comments Benchmarked**: `{len(raw_texts):,}`
- **Total Raw Tokens**: `{eval_metrics['total_raw_tokens']:,}`
- **Total Clean Tokens**: `{eval_metrics['total_clean_tokens']:,}` (`{eval_metrics['token_reduction_pct']}%` token reduction)
- **Raw Vocabulary Size**: `{eval_metrics['raw_vocabulary_size']:,}`
- **Clean Vocabulary Size**: `{eval_metrics['clean_vocabulary_size']:,}` (`{eval_metrics['vocabulary_reduction_pct']}%` vocabulary compression)
- **Single-Doc Latency**: `{bench_metrics['single_doc_latency_ms']} ms/doc`
- **Primary Generated Figures**:
  - `outputs/figures/preprocessing_comparison.png`
  - `outputs/figures/execution_time.png`
  - `outputs/figures/token_reduction.png`
  - `outputs/figures/vocabulary_reduction.png`

---

## 2. Pipeline Stage Execution Flow & Transformer Suite

1. **Validation** (`TextValidator`): Null, empty string, and Unicode boundary check.
2. **Lowercasing** (`LowercaseTransformer`): Case normalization preserving numbers & accents.
3. **Contraction Expansion** (`ContractionExpander`): `don't` $\\to$ `do not`, `can't` $\\to$ `cannot`.
4. **HTML Tag Removal** (`HTMLCleaner`): Strips `<p>`, `<div>`, `<br/>` tags & unescapes HTML entities.
5. **URL Removal** (`URLCleaner`): Strips `http://`, `https://`, `www.` or replaces with `[URL]`.
6. **Email Address Removal** (`EmailCleaner`): PII protection replacing emails with `[EMAIL]`.
7. **Emoji Processing** (`EmojiCleaner`): Converts 🤬 $\\to$ `:angry_face:` for sentiment retention.
8. **Number Normalization** (`NumberCleaner`): Replaces standalone numbers with `0`.
9. **Punctuation Removal** (`PunctuationCleaner`): Strips ASCII punctuation.
10. **Special Character Removal** (`SpecialCharacterCleaner`): Prunes non-linguistic noise.
11. **Whitespace Normalization** (`WhitespaceNormalizer`): Collapses tabs/newlines into clean single spaces.
12. **Stopword Removal** (`StopwordRemover`): Filters NLTK & Wikipedia domain noise (`talk`, `page`, `edit`).
13. **Lemmatization** (`Lemmatizer`): Maps words to base canonical dictionary lemmas via spaCy/WordNet.

---

## 3. Comparative Benchmarking & Performance Statistics

| Metric | Raw Text | Preprocessed Text | Delta / Reduction % |
| :--- | :--- | :--- | :--- |
| **Total Characters** | `{eval_metrics['total_raw_characters']:,}` | `{eval_metrics['total_clean_characters']:,}` | `{eval_metrics['character_reduction_pct']}%` |
| **Total Word Tokens** | `{eval_metrics['total_raw_tokens']:,}` | `{eval_metrics['total_clean_tokens']:,}` | `{eval_metrics['token_reduction_pct']}%` |
| **Unique Vocabulary** | `{eval_metrics['raw_vocabulary_size']:,}` | `{eval_metrics['clean_vocabulary_size']:,}` | `{eval_metrics['vocabulary_reduction_pct']}%` |
| **Mean Token Length** | `{eval_metrics['avg_raw_token_length']} words` | `{eval_metrics['avg_clean_token_length']} words` | Clean, concentrated signal |

---

## 4. Production Deployment & MLOps Integration Strategy

### 4.1 Deployment Architecture
1. **Real-time Inference API (FastAPI / Streamlit)**: Instantiate a lightweight `build_pipeline('traditional_ml')` singleton at app startup. Preprocess incoming raw JSON payload in < 2ms before calling model `predict_proba()`.
2. **Batch Ingestion Pipeline (Airflow / Spark)**: Use `transform_batch(texts, n_jobs=8)` to leverage multi-core CPU parallel processing for large-scale offline log processing.

### 4.2 Common Preprocessing Mistakes to Avoid
- **Blind Lowercasing for Cased Transformers**: Lowercasing text before passing to `bert-base-cased` destroys uppercase shouting signals (`SHUT UP`).
- **Indiscriminate Stopword Removal**: Removing stopwords (`"not"`) in sentiment models converts `"not toxic"` to `"toxic"`.
- **Order Inversion**: Running punctuation removal *before* contraction expansion causes `don't` $\\to$ `dont` (failing dictionary lookup).

---

## 5. Technical Interview Questions & Detailed Answers

### Q1: Why is pipeline stage ordering critical in text preprocessing?
**Answer**: Preprocessing operations are non-commutative. For example, running punctuation removal before contraction expansion transforms `"don't"` into `"dont"`, breaking standard contraction dictionary lookups. Running HTML cleaning first strips tags like `<a href="...">` before URL removal attempts to find web links.

### Q2: How do you configure preprocessing pipelines differently for TF-IDF vs Transformer models?
**Answer**:
- **TF-IDF**: Apply aggressive cleaning (lowercasing, stopword removal, lemmatization, punctuation stripping) to reduce sparse vocabulary dimensionality.
- **Transformers**: Apply minimal cleaning (normalize whitespace, strip HTML/URLs), retaining casing, punctuation, and stopwords, as self-attention mechanisms require full syntactic structure and subword tokenizers handle casing and vocabulary internally.
"""

with open(os.path.join(REPORTS_DIR, "preprocessing_report.md"), "w", encoding="utf-8") as f:
    f.write(report_md)

logger.info(f"Phase 3 Preprocessing Master Report exported to {REPORTS_DIR}/preprocessing_report.md")

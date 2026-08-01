"""
Script to generate all 15 Phase 2 Jupyter Notebooks (.ipynb) cleanly.
"""

import os
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

NOTEBOOKS_DIR = "notebooks"
os.makedirs(NOTEBOOKS_DIR, exist_ok=True)

notebook_configs = [
    {
        "filename": "01_dataset_overview.ipynb",
        "title": "Phase 2 - Step 6: Dataset Overview",
        "code": """import sys
sys.path.append("..")
import pandas as pd
from src.utils.data_loader import load_toxic_comment_data
from src.eda.dataset_overview import *

# 1. Load Dataset
df = load_toxic_comment_data()

# 2. Display Overview Metrics
shape = display_shape(df)
cols = display_columns(df)
dtypes = display_data_types(df)
info = display_dataset_info(df)
mem = display_memory_usage(df)
uniques = display_unique_value_counts(df)
samples = display_sample_records(df, n=5)

# 3. Generate Visualizations
plot_datatype_distribution(df)
plot_unique_value_counts(df)

# 4. Export Report
generate_dataset_overview_report(df)
print("Dataset Overview Completed Successfully!")
"""
    },
    {
        "filename": "02_missing_value_analysis.ipynb",
        "title": "Phase 2 - Step 7: Missing Value Analysis",
        "code": """import sys
sys.path.append("..")
import pandas as pd
from src.utils.data_loader import load_toxic_comment_data
from src.eda.missing_value_analysis import *

# 1. Load Dataset
df = load_toxic_comment_data()

# 2. Compute Missing Metrics
missing_counts = calculate_missing_values(df)
missing_pcts = calculate_missing_percentage(df)
summary_df = generate_missing_summary(df)

# 3. Generate Charts
plot_missing_bar_chart(df)
plot_missing_percentage(df)
plot_missing_heatmap(df)

# 4. Export Report
export_missing_report(df)
print("Missing Value Analysis Completed Successfully!")
"""
    },
    {
        "filename": "03_duplicate_value_analysis.ipynb",
        "title": "Phase 2 - Step 8: Duplicate Value Analysis",
        "code": """import sys
sys.path.append("..")
import pandas as pd
from src.utils.data_loader import load_toxic_comment_data
from src.eda.duplicate_analysis import *

# 1. Load Dataset
df = load_toxic_comment_data()

# 2. Duplicate Detection
dup_count = count_duplicate_rows(df, subset=["comment_text"])
dup_pct = calculate_duplicate_percentage(df, subset=["comment_text"])
comparison = compare_before_after_duplicates(df, subset=["comment_text"])

# 3. Generate Visualizations
plot_duplicate_count_bar(df)
plot_duplicate_percentage_pie(df)
plot_duplicate_summary_table(comparison)

# 4. Export Report
export_duplicate_report(df)
print("Duplicate Value Analysis Completed Successfully!")
"""
    },
    {
        "filename": "04_target_label_distribution.ipynb",
        "title": "Phase 2 - Step 9: Target Label Distribution Analysis",
        "code": """import sys
sys.path.append("..")
import pandas as pd
from src.utils.data_loader import load_toxic_comment_data
from src.eda.target_distribution import *

# 1. Load Dataset
df = load_toxic_comment_data()

# 2. Distribution Summary
summary_df = generate_distribution_summary(df)

# 3. Generate Charts
plot_label_distribution_bar(df)
plot_label_distribution_percentage(df)
plot_label_distribution_pie(df)
plot_target_distribution_table(summary_df)

# 4. Export Report
export_distribution_report(df)
print("Target Label Distribution Analysis Completed Successfully!")
"""
    },
    {
        "filename": "05_multilabel_cooccurrence_analysis.ipynb",
        "title": "Phase 2 - Step 10: Multi-Label Co-occurrence Analysis",
        "code": """import sys
sys.path.append("..")
import pandas as pd
from src.utils.data_loader import load_toxic_comment_data
from src.eda.multilabel_analysis import *

# 1. Load Dataset
df = load_toxic_comment_data()

# 2. Compute Multi-label Metrics
lbl_counts = calculate_labels_per_comment(df)
combos_df = calculate_label_combinations(df)
matrix = calculate_cooccurrence_matrix(df)
pairs_df = calculate_label_pair_frequency(df)

# 3. Generate Charts
plot_labels_per_comment_distribution(df)
plot_cooccurrence_heatmap(df)
plot_label_pair_frequency(df)
plot_label_combination_frequency(df)

# 4. Export Report
export_multilabel_report(df)
print("Multi-Label Co-occurrence Analysis Completed Successfully!")
"""
    },
    {
        "filename": "06_correlation_analysis.ipynb",
        "title": "Phase 2 - Step 11: Correlation Analysis",
        "code": """import sys
sys.path.append("..")
import pandas as pd
from src.utils.data_loader import load_toxic_comment_data
from src.eda.correlation_analysis import *

# 1. Load Dataset
df = load_toxic_comment_data()

# 2. Compute Matrices
p_corr = calculate_pearson_correlation(df)
s_corr = calculate_spearman_correlation(df)
k_corr = calculate_kendall_correlation(df)
comp_df = compare_correlation_methods(df)
summary = generate_correlation_summary(df)

# 3. Generate Visualizations
plot_correlation_heatmap(df)
plot_correlation_clustermap(df)
plot_correlation_network(df)

# 4. Export Report
export_correlation_report(df)
print("Correlation Analysis Completed Successfully!")
"""
    },
    {
        "filename": "07_comment_length_analysis.ipynb",
        "title": "Phase 2 - Step 12: Comment Length Analysis",
        "code": """import sys
sys.path.append("..")
import pandas as pd
from src.utils.data_loader import load_toxic_comment_data
from src.eda.comment_length_analysis import *

# 1. Load Dataset
df = load_toxic_comment_data()

# 2. Descriptive Stats
stats_dict = calculate_comment_statistics(df)
summary_df = summarize_comment_length(df)

# 3. Generate 300 DPI Figures
plot_length_histogram(df)
plot_length_kde(df)
plot_length_boxplot(df)
plot_length_violinplot(df)
compare_length_by_label(df)

# 4. Export Report
export_comment_length_report(df)
print("Comment Length Analysis Completed Successfully!")
"""
    },
    {
        "filename": "08_word_count_analysis.ipynb",
        "title": "Phase 2 - Step 13: Word Count Analysis",
        "code": """import sys
sys.path.append("..")
import pandas as pd
from src.utils.data_loader import load_toxic_comment_data
from src.eda.word_count_analysis import *

# 1. Load Dataset
df = load_toxic_comment_data()

# 2. Compute Statistics
stats_dict = calculate_word_statistics(df)
summary_df = summarize_word_count(df)

# 3. Generate 300 DPI Figures
plot_word_count_histogram(df)
plot_word_count_kde(df)
plot_word_count_boxplot(df)
plot_word_count_violinplot(df)
compare_word_count_by_label(df)
plot_word_count_distribution(df)

# 4. Export Report
export_word_count_report(df)
print("Word Count Analysis Completed Successfully!")
"""
    },
    {
        "filename": "09_character_count_analysis.ipynb",
        "title": "Phase 2 - Step 14: Character Count & Text Composition Analysis",
        "code": """import sys
sys.path.append("..")
import pandas as pd
from src.utils.data_loader import load_toxic_comment_data
from src.eda.character_analysis import *

# 1. Load Dataset
df = load_toxic_comment_data()

# 2. Character Composition Summary
summary = summarize_character_statistics(df)

# 3. Generate 300 DPI Figures
plot_character_distribution(df)

# 4. Export Report
export_character_analysis_report(df)
print("Character Count Analysis Completed Successfully!")
"""
    },
    {
        "filename": "10_sentence_length_analysis.ipynb",
        "title": "Phase 2 - Step 15: Sentence Length Analysis",
        "code": """import sys
sys.path.append("..")
import pandas as pd
from src.utils.data_loader import load_toxic_comment_data
from src.eda.sentence_length_analysis import *

# 1. Load Dataset
df = load_toxic_comment_data()

# 2. Compute Sentence Stats
stats_dict = calculate_sentence_statistics(df)
summary_df = summarize_sentence_analysis(df)
longest_sent, _ = identify_longest_sentence(df)
shortest_sent, _ = identify_shortest_sentence(df)

# 3. Generate 300 DPI Figures
plot_sentence_distribution(df)
plot_sentence_boxplot(df)
plot_sentence_violinplot(df)
compare_sentence_length_by_label(df)

# 4. Export Report
export_sentence_report(df)
print("Sentence Length Analysis Completed Successfully!")
"""
    },
    {
        "filename": "11_word_frequency_analysis.ipynb",
        "title": "Phase 2 - Step 16: Word Frequency Analysis",
        "code": """import sys
sys.path.append("..")
import pandas as pd
from src.utils.data_loader import load_toxic_comment_data
from src.eda.word_frequency_analysis import *

# 1. Load Dataset
df = load_toxic_comment_data()

# 2. Vocabulary & Word Frequencies
tokens = preprocess_for_frequency(df)
vocab = calculate_vocabulary_size(tokens)
rare = calculate_rare_words(tokens, threshold=5)
top20 = calculate_word_frequency(tokens, top_n=20)
label_freqs = calculate_label_word_frequency(df, top_n=10)

# 3. Generate 300 DPI Figures
plot_top_words(top20, top_n=20)
plot_label_top_words(label_freqs)
plot_frequency_distributions(tokens)

# 4. Export Report
export_frequency_report(df)
print("Word Frequency Analysis Completed Successfully!")
"""
    },
    {
        "filename": "12_word_cloud_analysis.ipynb",
        "title": "Phase 2 - Step 17: Word Cloud Analysis",
        "code": """import sys
sys.path.append("..")
import pandas as pd
from src.utils.data_loader import load_toxic_comment_data
from src.eda.word_cloud_analysis import *

# 1. Load Dataset
df = load_toxic_comment_data()

# 2. Generate Figures & Export Report
export_wordcloud_report(df)
print("Word Cloud Analysis Completed Successfully!")
"""
    },
    {
        "filename": "13_bigram_analysis.ipynb",
        "title": "Phase 2 - Step 18: Bigram Analysis",
        "code": """import sys
sys.path.append("..")
import pandas as pd
from src.utils.data_loader import load_toxic_comment_data
from src.eda.bigram_analysis import *

# 1. Load Dataset
df = load_toxic_comment_data()

# 2. Generate Figures & Export Report
export_bigram_report(df)
print("Bigram Analysis Completed Successfully!")
"""
    },
    {
        "filename": "14_trigram_analysis.ipynb",
        "title": "Phase 2 - Step 19: Trigram Analysis",
        "code": """import sys
sys.path.append("..")
import pandas as pd
from src.utils.data_loader import load_toxic_comment_data
from src.eda.trigram_analysis import *

# 1. Load Dataset
df = load_toxic_comment_data()

# 2. Generate Figures & Export Report
export_trigram_report(df)
print("Trigram Analysis Completed Successfully!")
"""
    },
    {
        "filename": "15_eda_summary_report.ipynb",
        "title": "Phase 2 - Step 20: Enterprise EDA Summary Report",
        "code": """import sys
sys.path.append("..")
import pandas as pd
from src.utils.data_loader import load_toxic_comment_data
from src.reports.eda_summary import *

# 1. Load Dataset
df = load_toxic_comment_data()

# 2. Generate Executive Dashboard Figure
generate_executive_dashboard(df)

# 3. Export Markdown Report
export_markdown_report(df)

# 4. Export PDF Report
export_pdf_report()

print("Master Enterprise EDA Summary Report Completed Successfully!")
"""
    },
]

for cfg in notebook_configs:
    filepath = os.path.join(NOTEBOOKS_DIR, cfg["filename"])
    nb_json = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [f"# {cfg['title']}\n"]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": cfg["code"].splitlines(keepends=True)
            }
        ],
        "metadata": {
            "language_info": {"name": "python"}
        },
        "nbformat": 4,
        "nbformat_minor": 2
    }
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(nb_json, f, indent=1)
    logger.info(f"Generated notebook {filepath}")

print("All 15 notebooks created successfully!")

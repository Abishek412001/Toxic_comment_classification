"""
Master pipeline execution script to run all 15 Phase 2 EDA steps end-to-end.
"""

import sys
import os
sys.path.insert(0, os.path.abspath("."))
import logging
import pandas as pd

from src.utils.data_loader import load_toxic_comment_data
from src.eda.dataset_overview import display_shape, plot_datatype_distribution, plot_unique_value_counts, generate_dataset_overview_report
from src.eda.missing_value_analysis import export_missing_report, plot_missing_bar_chart, plot_missing_percentage, plot_missing_heatmap
from src.eda.duplicate_analysis import export_duplicate_report, plot_duplicate_count_bar, plot_duplicate_percentage_pie, plot_duplicate_summary_table, compare_before_after_duplicates
from src.eda.target_distribution import export_distribution_report, plot_label_distribution_bar, plot_label_distribution_percentage, plot_label_distribution_pie, plot_target_distribution_table, generate_distribution_summary
from src.eda.multilabel_analysis import export_multilabel_report, plot_labels_per_comment_distribution, plot_cooccurrence_heatmap, plot_label_pair_frequency, plot_label_combination_frequency
from src.eda.correlation_analysis import export_correlation_report, plot_correlation_heatmap, plot_correlation_clustermap, plot_correlation_network
from src.eda.comment_length_analysis import export_comment_length_report, plot_length_histogram, plot_length_kde, plot_length_boxplot, plot_length_violinplot, compare_length_by_label
from src.eda.word_count_analysis import export_word_count_report, plot_word_count_histogram, plot_word_count_kde, plot_word_count_boxplot, plot_word_count_violinplot, compare_word_count_by_label, plot_word_count_distribution
from src.eda.character_analysis import export_character_analysis_report, plot_character_distribution
from src.eda.sentence_length_analysis import export_sentence_report, plot_sentence_distribution, plot_sentence_boxplot, plot_sentence_violinplot, compare_sentence_length_by_label
from src.eda.word_frequency_analysis import export_frequency_report, preprocess_for_frequency, calculate_word_frequency, calculate_label_word_frequency, plot_top_words, plot_label_top_words, plot_frequency_distributions
from src.eda.word_cloud_analysis import export_wordcloud_report
from src.eda.bigram_analysis import export_bigram_report
from src.eda.trigram_analysis import export_trigram_report
from src.reports.eda_summary import generate_executive_dashboard, export_markdown_report, export_pdf_report

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def main():
    logger.info("Starting Master Phase 2 Execution Pipeline...")

    # Load Dataset
    df = load_toxic_comment_data()

    # Step 6: Dataset Overview
    logger.info("Running Step 6: Dataset Overview...")
    display_shape(df)
    plot_datatype_distribution(df)
    plot_unique_value_counts(df)
    generate_dataset_overview_report(df)

    # Step 7: Missing Values
    logger.info("Running Step 7: Missing Value Analysis...")
    plot_missing_bar_chart(df)
    plot_missing_percentage(df)
    plot_missing_heatmap(df)
    export_missing_report(df)

    # Step 8: Duplicates
    logger.info("Running Step 8: Duplicate Value Analysis...")
    comp = compare_before_after_duplicates(df, subset=["comment_text"])
    plot_duplicate_count_bar(df)
    plot_duplicate_percentage_pie(df)
    plot_duplicate_summary_table(comp)
    export_duplicate_report(df)

    # Step 9: Target Label Distribution
    logger.info("Running Step 9: Target Label Distribution Analysis...")
    summary_dist = generate_distribution_summary(df)
    plot_label_distribution_bar(df)
    plot_label_distribution_percentage(df)
    plot_label_distribution_pie(df)
    plot_target_distribution_table(summary_dist)
    export_distribution_report(df)

    # Step 10: Multi-Label Co-occurrence
    logger.info("Running Step 10: Multi-Label Co-occurrence Analysis...")
    plot_labels_per_comment_distribution(df)
    plot_cooccurrence_heatmap(df)
    plot_label_pair_frequency(df)
    plot_label_combination_frequency(df)
    export_multilabel_report(df)

    # Step 11: Correlation
    logger.info("Running Step 11: Correlation Analysis...")
    plot_correlation_heatmap(df)
    plot_correlation_clustermap(df)
    plot_correlation_network(df)
    export_correlation_report(df)

    # Step 12: Comment Length
    logger.info("Running Step 12: Comment Length Analysis...")
    plot_length_histogram(df)
    plot_length_kde(df)
    plot_length_boxplot(df)
    plot_length_violinplot(df)
    compare_length_by_label(df)
    export_comment_length_report(df)

    # Step 13: Word Count
    logger.info("Running Step 13: Word Count Analysis...")
    plot_word_count_histogram(df)
    plot_word_count_kde(df)
    plot_word_count_boxplot(df)
    plot_word_count_violinplot(df)
    compare_word_count_by_label(df)
    plot_word_count_distribution(df)
    export_word_count_report(df)

    # Step 14: Character Count & Composition
    logger.info("Running Step 14: Character Count Analysis...")
    plot_character_distribution(df)
    export_character_analysis_report(df)

    # Step 15: Sentence Length
    logger.info("Running Step 15: Sentence Length Analysis...")
    plot_sentence_distribution(df)
    plot_sentence_boxplot(df)
    plot_sentence_violinplot(df)
    compare_sentence_length_by_label(df)
    export_sentence_report(df)

    # Step 16: Word Frequency
    logger.info("Running Step 16: Word Frequency Analysis...")
    tokens = preprocess_for_frequency(df)
    top20 = calculate_word_frequency(tokens, top_n=20)
    label_freqs = calculate_label_word_frequency(df, top_n=10)
    plot_top_words(top20, top_n=20)
    plot_label_top_words(label_freqs)
    plot_frequency_distributions(tokens)
    export_frequency_report(df)

    # Step 17: Word Clouds
    logger.info("Running Step 17: Word Cloud Analysis...")
    export_wordcloud_report(df)

    # Step 18: Bigram Analysis
    logger.info("Running Step 18: Bigram Analysis...")
    export_bigram_report(df)

    # Step 19: Trigram Analysis
    logger.info("Running Step 19: Trigram Analysis...")
    export_trigram_report(df)

    # Step 20: Enterprise EDA Summary Report
    logger.info("Running Step 20: Enterprise EDA Summary Report...")
    generate_executive_dashboard(df)
    export_markdown_report(df)
    export_pdf_report()

    logger.info("ALL 15 PHASE 2 STEPS COMPLETED & VERIFIED SUCCESSFULLY!")

if __name__ == "__main__":
    main()

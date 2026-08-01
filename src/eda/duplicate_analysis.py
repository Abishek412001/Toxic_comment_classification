"""
Duplicate Value Analysis Module.

Provides modular functions to detect full record duplicates, text-only duplicates,
label-conflicting duplicates, generate figures, and export reports.
"""

import os
import logging
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def count_duplicate_rows(df: pd.DataFrame, subset: Optional[List[str]] = None) -> int:
    """Counts the number of duplicate rows.

    Args:
        df: Input DataFrame.
        subset: Optional list of column names to evaluate duplicates on.

    Returns:
        Integer count of duplicate rows.
    """
    if df is None or not isinstance(df, pd.DataFrame):
        raise ValueError("Input must be a valid pandas DataFrame.")

    dup_count = int(df.duplicated(subset=subset, keep="first").sum())
    logger.info(f"Duplicate count (subset={subset}): {dup_count}")
    return dup_count


def identify_duplicate_records(
    df: pd.DataFrame, subset: Optional[List[str]] = None
) -> pd.DataFrame:
    """Retrieves all duplicate records (including initial occurrence).

    Args:
        df: Input DataFrame.
        subset: Optional list of column names to evaluate duplicates on.

    Returns:
        pd.DataFrame containing marked duplicate rows.
    """
    if df is None or not isinstance(df, pd.DataFrame):
        raise ValueError("Input must be a valid pandas DataFrame.")

    duplicates_df = df[df.duplicated(subset=subset, keep=False)].copy()
    logger.info(f"Identified {len(duplicates_df)} total duplicate occurrences.")
    return duplicates_df


def calculate_duplicate_percentage(
    df: pd.DataFrame, subset: Optional[List[str]] = None
) -> float:
    """Calculates percentage of duplicate rows.

    Args:
        df: Input DataFrame.
        subset: Optional list of column names to evaluate duplicates on.

    Returns:
        Float percentage (0-100).
    """
    if df is None or not isinstance(df, pd.DataFrame):
        raise ValueError("Input must be a valid pandas DataFrame.")

    if len(df) == 0:
        return 0.0

    dup_count = count_duplicate_rows(df, subset=subset)
    dup_pct = (dup_count / len(df)) * 100.0
    logger.info(f"Duplicate percentage: {dup_pct:.4f}%")
    return round(dup_pct, 4)


def compare_before_after_duplicates(
    df: pd.DataFrame, subset: Optional[List[str]] = None
) -> Dict[str, Any]:
    """Computes comparison metrics before and after deduplication.

    Args:
        df: Input DataFrame.
        subset: Columns to evaluate duplicates on.

    Returns:
        Dict containing before/after shapes, rows removed, and memory savings.
    """
    if df is None or not isinstance(df, pd.DataFrame):
        raise ValueError("Input must be a valid pandas DataFrame.")

    before_rows = len(df)
    before_mem_mb = round(df.memory_usage(deep=True).sum() / (1024 * 1024), 4)

    dedup_df = remove_duplicate_records(df, keep="first", subset=subset)
    after_rows = len(dedup_df)
    after_mem_mb = round(dedup_df.memory_usage(deep=True).sum() / (1024 * 1024), 4)

    rows_removed = before_rows - after_rows
    mem_saved_mb = round(before_mem_mb - after_mem_mb, 4)

    comparison = {
        "before_rows": before_rows,
        "after_rows": after_rows,
        "rows_removed": rows_removed,
        "duplicate_percentage": round((rows_removed / before_rows) * 100.0, 4) if before_rows > 0 else 0.0,
        "before_memory_mb": before_mem_mb,
        "after_memory_mb": after_mem_mb,
        "memory_saved_mb": mem_saved_mb,
    }
    logger.info(f"Deduplication comparison: {comparison}")
    return comparison


def remove_duplicate_records(
    df: pd.DataFrame, keep: str = "first", subset: Optional[List[str]] = None
) -> pd.DataFrame:
    """Removes duplicate rows from DataFrame.

    Args:
        df: Input DataFrame.
        keep: Which occurrence to keep ('first', 'last', False).
        subset: Optional columns to evaluate duplicates on.

    Returns:
        Deduplicated pd.DataFrame.
    """
    if df is None or not isinstance(df, pd.DataFrame):
        raise ValueError("Input must be a valid pandas DataFrame.")

    dedup_df = df.drop_duplicates(subset=subset, keep=keep).reset_index(drop=True)
    logger.info(f"Removed duplicates. New shape: {dedup_df.shape}")
    return dedup_df


def plot_duplicate_count_bar(
    df: pd.DataFrame, output_path: str = "outputs/figures/duplicate_count_bar.png"
) -> None:
    """Plots bar chart of Unique vs Duplicate records.

    Args:
        df: Input DataFrame.
        output_path: Output figure file path.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    dup_count = count_duplicate_rows(df, subset=["comment_text"])
    unique_count = len(df) - dup_count

    data = pd.DataFrame({"Category": ["Unique Comments", "Duplicate Comments"], "Count": [unique_count, dup_count]})

    plt.figure(figsize=(8, 5))
    ax = sns.barplot(x="Category", y="Count", data=data, palette=["#2ecc71", "#e74c3c"])
    plt.title("Unique vs Duplicate Comments Count", fontsize=14, fontweight="bold", pad=12)
    plt.ylabel("Record Count", fontsize=12, labelpad=8)
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    for p in ax.patches:
        val = int(p.get_height())
        ax.annotate(
            f"{val:,}",
            (p.get_x() + p.get_width() / 2.0, val),
            ha="center",
            va="bottom",
            fontsize=11,
            xytext=(0, 3),
            textcoords="offset points",
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    logger.info(f"Saved duplicate count bar chart to {output_path}")


def plot_duplicate_percentage_pie(
    df: pd.DataFrame, output_path: str = "outputs/figures/duplicate_percentage_pie.png"
) -> None:
    """Plots pie chart of Duplicate vs Unique record percentages.

    Args:
        df: Input DataFrame.
        output_path: Output figure file path.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    dup_count = count_duplicate_rows(df, subset=["comment_text"])
    unique_count = len(df) - dup_count

    plt.figure(figsize=(7, 7))
    plt.pie(
        [unique_count, max(dup_count, 1e-5)],
        labels=["Unique Records", "Duplicate Records"],
        autopct="%1.2f%%",
        colors=["#3498db", "#e74c3c"],
        explode=[0, 0.1] if dup_count > 0 else [0, 0],
        startangle=140,
        textprops={"fontsize": 12},
    )
    plt.title("Dataset Proportion: Unique vs Duplicates", fontsize=14, fontweight="bold", pad=12)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    logger.info(f"Saved duplicate percentage pie chart to {output_path}")


def plot_duplicate_summary_table(
    comparison: Dict[str, Any], output_path: str = "outputs/figures/duplicate_summary_table.png"
) -> None:
    """Renders a high-res summary table image of before vs after deduplication.

    Args:
        comparison: Comparison dictionary.
        output_path: Output figure file path.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 3))
    ax.axis("tight")
    ax.axis("off")

    table_data = [
        ["Metric", "Before Deduplication", "After Deduplication", "Difference"],
        ["Total Rows", f"{comparison['before_rows']:,}", f"{comparison['after_rows']:,}", f"-{comparison['rows_removed']:,}"],
        ["Memory Usage (MB)", f"{comparison['before_memory_mb']} MB", f"{comparison['after_memory_mb']} MB", f"-{comparison['memory_saved_mb']} MB"],
        ["Duplicate Percentage", f"{comparison['duplicate_percentage']:.2f}%", "0.00%", f"-{comparison['duplicate_percentage']:.2f}%"],
    ]

    table = ax.table(cellText=table_data, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)

    for i in range(4):
        table[(0, i)].set_facecolor("#2c3e50")
        table[(0, i)].get_text().set_color("white")
        table[(0, i)].get_text().set_weight("bold")

    plt.title("Dataset Deduplication Impact Summary", fontsize=14, fontweight="bold", pad=15)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    logger.info(f"Saved duplicate summary table figure to {output_path}")


def export_duplicate_report(
    df: pd.DataFrame, report_path: str = "outputs/reports/duplicate_analysis_report.md"
) -> None:
    """Exports Duplicate Value Analysis Markdown report.

    Args:
        df: Input DataFrame.
        report_path: Path to write markdown report.
    """
    os.makedirs(os.path.dirname(report_path), exist_ok=True)

    full_dups = count_duplicate_rows(df, subset=None)
    text_dups = count_duplicate_rows(df, subset=["comment_text"])

    comp_full = compare_before_after_duplicates(df, subset=None)
    comp_text = compare_before_after_duplicates(df, subset=["comment_text"])

    # Detect label-conflicting duplicates (same comment, different toxicity labels)
    text_grouped = df.groupby("comment_text")
    label_cols = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
    label_conflicts = 0
    if len(df) > 0 and all(c in df.columns for c in label_cols):
        label_std = text_grouped[label_cols].nunique()
        label_conflicts = int((label_std > 1).any(axis=1).sum())

    report_md = f"""# Toxic Comment Classification - Duplicate Value Analysis Report

## 1. Executive Summary & Overview Metrics

- **Total Initial Rows**: `{len(df):,}`
- **Exact Full-Record Duplicates**: `{full_dups:,}` (`{comp_full['duplicate_percentage']:.2f}%`)
- **Comment Text-Only Duplicates**: `{text_dups:,}` (`{comp_text['duplicate_percentage']:.2f}%`)
- **Label-Conflicting Comment Duplicates**: `{label_conflicts:,}`
- **Dataset Size Before Cleaning**: `{comp_text['before_rows']:,}` rows (`{comp_text['before_memory_mb']} MB`)
- **Dataset Size After Deduplication**: `{comp_text['after_rows']:,}` rows (`{comp_text['after_memory_mb']} MB`)
- **Net Rows Removed**: `{comp_text['rows_removed']:,}`

---

## 2. Detailed Duplicate Breakdown

| Duplicate Category | Count | Percentage (%) | Severity | Primary Risk |
| :--- | :--- | :--- | :--- | :--- |
| **Exact Record Duplicates** (All Columns Match) | `{full_dups:,}` | `{comp_full['duplicate_percentage']:.2f}%` | Low | Memory waste & gradient over-weighting |
| **Comment Text Duplicates** (Text Matches) | `{text_dups:,}` | `{comp_text['duplicate_percentage']:.2f}%` | High | Train/Test Data Leakage |
| **Label-Conflicting Duplicates** (Same Text, Diff Labels) | `{label_conflicts:,}` | `{(label_conflicts/len(df))*100 if len(df)>0 else 0:.2f}%` | Critical | Model training instability & loss divergence |

---

## 3. Visualization Callouts & Impact Analysis

### Figure 1: Duplicate vs Unique Bar Chart (`outputs/figures/duplicate_count_bar.png`)
- **Business Insight**: Quantifies repeated spam or automated copy-paste comments across user forums.
- **Technical Insight**: Visualizes dataset uniqueness ratio to validate deduplication priority.
- **Impact on ML Models**: Prevents identical comments from appearing in both train and validation splits.
- **Recommended Action**: Deduplicate dataset on `comment_text` keeping the first occurrence.

### Figure 2: Duplicate Percentage Pie Chart (`outputs/figures/duplicate_percentage_pie.png`)
- **Business Insight**: Establishes data quality compliance percentage for corporate audit reporting.
- **Technical Insight**: Displays overall data retention ratio after cleaning.
- **Impact on ML Models**: Ensures clean evaluation metrics on truly independent evaluation sets.
- **Recommended Action**: Maintain automated deduplication filters in streaming production data pipelines.

### Figure 3: Deduplication Impact Summary (`outputs/figures/duplicate_summary_table.png`)
- **Business Insight**: Demonstrates compute resource optimization and infrastructure savings.
- **Technical Insight**: Tracks exact RAM memory savings and row reduction post-cleaning.
- **Impact on ML Models**: Accelerates training convergence by removing redundant backward-pass computations.
- **Recommended Action**: Retain deduplicated dataset for all downstream feature engineering stages.

---

## 4. Deep-Dive Interpretations & Best Practices

### Business Interpretation
Deduplication eliminates repeated automated bot spam and copy-paste text, preventing the moderation engine from being biased toward high-frequency spam templates.

### Technical Interpretation
Deduplicating on `comment_text` prevents **Data Leakage** between training and validation folds during K-Fold cross-validation.

### Recommendations
1. **Deduplication Strategy**: Execute `df.drop_duplicates(subset=['comment_text'], keep='first')` prior to train-test splitting.
2. **Conflict Resolution**: If label-conflicting duplicates exist, aggregate label targets using `max()` or majority voting.

---

## 5. Industry Best Practices & Technical Foundations

### Why Duplicate Analysis is Important
In NLP models, duplicate text entries between training and evaluation splits cause severe evaluation bias, where models achieve artificially inflated accuracy on memorized text while failing on unseen production traffic.

### Types of Duplicates
1. **Exact Duplicates**: Identical raw bytes and labels.
2. **Text-Only Duplicates**: Identical string text with potentially conflicting label assignments.
3. **Near-Duplicates / Semantic Duplicates**: Paraphrased text or character-level variations (e.g. typos, added spaces).

### Interview Q&A

#### Q1: What is Data Leakage, and how do duplicate records cause it in NLP pipelines?
**Answer**: Data Leakage occurs when information from outside the training dataset is used to train the model. When duplicate comments exist across train and test sets, the model memorizes specific text strings rather than learning generalizable semantic patterns, causing overfitted evaluation scores.

#### Q2: How should label-conflicting duplicates (same text, different target labels) be handled?
**Answer**: Label-conflicting duplicates occur due to human annotator disagreement. They can be resolved by:
- Taking the `max()` logical OR across multi-hot targets (conservative safety approach).
- Applying majority voting or soft probabilistic targets ($y \in [0, 1]$).
- Dropping conflicting samples if annotation noise is unresolvable.
"""

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_md)

    logger.info(f"Duplicate Value Analysis Report exported to {report_path}")

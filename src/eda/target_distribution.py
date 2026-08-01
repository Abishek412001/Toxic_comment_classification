"""
Target Label Distribution Analysis Module.

Provides production-grade functions to compute multi-label target counts, percentages,
imbalance ratios, generate publication-quality figures, and export markdown reports.
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

DEFAULT_LABELS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]


def calculate_label_counts(df: pd.DataFrame, label_cols: Optional[List[str]] = None) -> pd.DataFrame:
    """Calculates positive and negative sample counts for each target label.

    Args:
        df: Input DataFrame.
        label_cols: List of label column names.

    Returns:
        pd.DataFrame containing positive and negative counts per label.
    """
    if df is None or not isinstance(df, pd.DataFrame):
        raise ValueError("Input must be a valid pandas DataFrame.")

    cols = label_cols or DEFAULT_LABELS
    pos_counts = df[cols].sum(axis=0)
    neg_counts = len(df) - pos_counts

    result_df = pd.DataFrame(
        {
            "Label": cols,
            "Positive Count": pos_counts.values,
            "Negative Count": neg_counts.values,
        }
    )
    logger.info(f"Calculated label counts:\n{result_df}")
    return result_df


def calculate_label_percentages(df: pd.DataFrame, label_cols: Optional[List[str]] = None) -> pd.DataFrame:
    """Calculates positive percentage and imbalance ratio for each label.

    Args:
        df: Input DataFrame.
        label_cols: List of label column names.

    Returns:
        pd.DataFrame containing percentage and imbalance ratio (Neg:Pos).
    """
    if df is None or not isinstance(df, pd.DataFrame):
        raise ValueError("Input must be a valid pandas DataFrame.")

    total_rows = len(df)
    counts_df = calculate_label_counts(df, label_cols=label_cols)

    counts_df["Positive Percentage (%)"] = (counts_df["Positive Count"] / total_rows * 100.0).round(4)
    counts_df["Negative Percentage (%)"] = (counts_df["Negative Count"] / total_rows * 100.0).round(4)

    # Imbalance Ratio = Neg / Pos
    counts_df["Imbalance Ratio (Neg:Pos)"] = np.where(
        counts_df["Positive Count"] > 0,
        (counts_df["Negative Count"] / counts_df["Positive Count"]).round(2),
        np.nan,
    )

    logger.info("Calculated label percentages and imbalance ratios.")
    return counts_df


def generate_distribution_summary(df: pd.DataFrame, label_cols: Optional[List[str]] = None) -> pd.DataFrame:
    """Generates ranked summary table of target labels.

    Args:
        df: Input DataFrame.
        label_cols: Target label columns.

    Returns:
        pd.DataFrame summary sorted by positive frequency.
    """
    if df is None or not isinstance(df, pd.DataFrame):
        raise ValueError("Input must be a valid pandas DataFrame.")

    df_pct = calculate_label_percentages(df, label_cols=label_cols)
    df_summary = df_pct.sort_values(by="Positive Count", ascending=False).reset_index(drop=True)
    df_summary["Rank"] = range(1, len(df_summary) + 1)

    logger.info("Generated ranked distribution summary.")
    return df_summary


def plot_label_distribution_bar(
    df: pd.DataFrame, label_cols: Optional[List[str]] = None, output_path: str = "outputs/figures/target_distribution_bar.png"
) -> None:
    """Plots positive label count bar chart annotated with exact values.

    Args:
        df: Input DataFrame.
        label_cols: Target label columns.
        output_path: Target figure path.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    summary = generate_distribution_summary(df, label_cols=label_cols)

    plt.figure(figsize=(10, 5))
    ax = sns.barplot(x="Label", y="Positive Count", data=summary, palette="magma")
    plt.title("Target Label Positive Sample Counts", fontsize=14, fontweight="bold", pad=12)
    plt.xlabel("Target Toxicity Label", fontsize=12, labelpad=8)
    plt.ylabel("Positive Sample Count", fontsize=12, labelpad=8)
    plt.xticks(rotation=30, ha="right")
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    for p in ax.patches:
        val = int(p.get_height())
        ax.annotate(
            f"{val:,}",
            (p.get_x() + p.get_width() / 2.0, val),
            ha="center",
            va="bottom",
            fontsize=10,
            xytext=(0, 3),
            textcoords="offset points",
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    logger.info(f"Saved label count bar chart to {output_path}")


def plot_label_distribution_percentage(
    df: pd.DataFrame, label_cols: Optional[List[str]] = None, output_path: str = "outputs/figures/target_distribution_percentage.png"
) -> None:
    """Plots positive label percentage bar chart with imbalance callouts.

    Args:
        df: Input DataFrame.
        label_cols: Target label columns.
        output_path: Target figure path.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    summary = generate_distribution_summary(df, label_cols=label_cols)

    plt.figure(figsize=(10, 5))
    ax = sns.barplot(x="Label", y="Positive Percentage (%)", data=summary, palette="viridis")
    plt.title("Target Label Positive Percentage (%)", fontsize=14, fontweight="bold", pad=12)
    plt.xlabel("Target Toxicity Label", fontsize=12, labelpad=8)
    plt.ylabel("Percentage of Total Dataset (%)", fontsize=12, labelpad=8)
    plt.xticks(rotation=30, ha="right")
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    for p in ax.patches:
        val = float(p.get_height())
        ax.annotate(
            f"{val:.2f}%",
            (p.get_x() + p.get_width() / 2.0, val),
            ha="center",
            va="bottom",
            fontsize=10,
            xytext=(0, 3),
            textcoords="offset points",
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    logger.info(f"Saved label percentage bar chart to {output_path}")


def plot_label_distribution_pie(
    df: pd.DataFrame, label_cols: Optional[List[str]] = None, output_path: str = "outputs/figures/target_distribution_pie.png"
) -> None:
    """Plots 2x3 grid of pie charts showing Positive vs Negative proportion per label.

    Args:
        df: Input DataFrame.
        label_cols: Target label columns.
        output_path: Target figure path.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cols = label_cols or DEFAULT_LABELS
    counts_df = calculate_label_counts(df, label_cols=cols)

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes = axes.flatten()

    for idx, row in counts_df.iterrows():
        lbl = row["Label"]
        pos = row["Positive Count"]
        neg = row["Negative Count"]
        ax = axes[idx]

        ax.pie(
            [neg, max(pos, 1e-5)],
            labels=["Non-Toxic", "Positive"],
            autopct="%1.1f%%",
            colors=["#2ecc71", "#e74c3c"],
            startangle=140,
            textprops={"fontsize": 9},
        )
        ax.set_title(f"Label: {lbl}", fontsize=11, fontweight="bold")

    plt.suptitle("Positive vs Negative Proportion per Toxic Category", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    logger.info(f"Saved label distribution pie grid to {output_path}")


def plot_target_distribution_table(
    summary_df: pd.DataFrame, output_path: str = "outputs/figures/target_distribution_table.png"
) -> None:
    """Plots summary table image of target distributions.

    Args:
        summary_df: Summary DataFrame.
        output_path: Target figure path.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis("tight")
    ax.axis("off")

    table_data = [
        ["Rank", "Label", "Positive Count", "Negative Count", "Pos %", "Imbalance Ratio (Neg:Pos)"],
    ]
    for _, row in summary_df.iterrows():
        table_data.append(
            [
                str(row["Rank"]),
                row["Label"],
                f"{int(row['Positive Count']):,}",
                f"{int(row['Negative Count']):,}",
                f"{row['Positive Percentage (%)']:.2f}%",
                f"{row['Imbalance Ratio (Neg:Pos)']}:1",
            ]
        )

    table = ax.table(cellText=table_data, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.1, 1.6)

    for i in range(6):
        table[(0, i)].set_facecolor("#2c3e50")
        table[(0, i)].get_text().set_color("white")
        table[(0, i)].get_text().set_weight("bold")

    plt.title("Target Label Distribution & Imbalance Summary Table", fontsize=14, fontweight="bold", pad=15)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    logger.info(f"Saved target distribution table image to {output_path}")


def export_distribution_report(
    df: pd.DataFrame, label_cols: Optional[List[str]] = None, report_path: str = "outputs/reports/target_distribution_report.md"
) -> None:
    """Exports Target Label Distribution Analysis Markdown report.

    Args:
        df: Input DataFrame.
        label_cols: Target label columns.
        report_path: Target report path.
    """
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    summary = generate_distribution_summary(df, label_cols=label_cols)

    most_common = summary.iloc[0]["Label"]
    least_common = summary.iloc[-1]["Label"]

    report_md = f"""# Toxic Comment Classification - Target Label Distribution Analysis Report

## 1. Executive Summary & Overview Metrics

- **Dataset Name**: Toxic Comment Classification
- **Total Comments Analyzed**: `{len(df):,}`
- **Target Label Count**: `{len(summary)}`
- **Most Common Label**: `{most_common}` (`{summary.iloc[0]['Positive Percentage (%)']:.2f}%` positive)
- **Least Common Label**: `{least_common}` (`{summary.iloc[-1]['Positive Percentage (%)']:.2f}%` positive)
- **Imbalance Range**: `{summary.iloc[0]['Imbalance Ratio (Neg:Pos)']}:1` (Most Frequent) to `{summary.iloc[-1]['Imbalance Ratio (Neg:Pos)']}:1` (Least Frequent)

---

## 2. Tabular Target Label Distribution Summary

| Rank | Label Name | Positive Count | Negative Count | Pos Percentage (%) | Imbalance Ratio (Neg:Pos) | Frequency Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
"""
    for _, row in summary.iterrows():
        ratio = row['Imbalance Ratio (Neg:Pos)']
        status = "MAJORITY TOXIC" if row['Label'] == 'toxic' else ("EXTREME IMBALANCE" if ratio > 50 else "MODERATE IMBALANCE")
        report_md += f"| `{row['Rank']}` | `{row['Label']}` | `{int(row['Positive Count']):,}` | `{int(row['Negative Count']):,}` | `{row['Positive Percentage (%)']:.2f}%` | `{ratio}:1` | `{status}` |\n"

    report_md += """

---

## 3. Visualization Callouts & Impact Analysis

### Figure 1: Positive Label Count Bar Chart (`outputs/figures/target_distribution_bar.png`)
- **Business Insight**: Reveals operational workload per toxicity class for human moderation queues.
- **Technical Insight**: Establishes class hierarchy; general `toxic` comments occur ~10x more frequently than niche categories like `threat`.
- **Impact on ML**: Standard accuracy will be deceptively high (>90%) by predicting all zeros.
- **Potential Risks**: Model will completely fail to detect rare severe categories (`threat`, `identity_hate`).
- **Recommended Action**: Implement Focal Loss, BCE with `pos_weight`, or Class-Weighted loss functions.

### Figure 2: Positive Label Percentage Bar Chart (`outputs/figures/target_distribution_percentage.png`)
- **Business Insight**: Communicates relative risk profile of online discourse across platform channels.
- **Technical Insight**: Quantifies severe positive scarcity across secondary toxicity tags (< 1% for `threat`).
- **Impact on ML**: Requires Precision-Recall AUC (PR-AUC) and F1-Score instead of ROC-AUC or Raw Accuracy.
- **Potential Risks**: High false negative rate for high-harm safety violations.
- **Recommended Action**: Tune decision thresholds per label independently using validation PR curves.

### Figure 3: Label Proportion Pie Charts (`outputs/figures/target_distribution_pie.png`)
- **Business Insight**: Visualizes extreme asymmetry between benign and abusive comments.
- **Technical Insight**: Confirms heavy negative majority across all 6 binary targets.
- **Impact on ML**: Gradients will be dominated by negative samples during backpropagation.
- **Potential Risks**: Model weights converge toward low-variance constant predictions.
- **Recommended Action**: Use hard negative mining or stratify train-validation splits using Iterative Stratification.

### Figure 4: Ranked Summary Table (`outputs/figures/target_distribution_table.png`)
- **Business Insight**: Provides executive tabular overview for SLA planning.
- **Technical Insight**: Ranks labels by positive frequency to structure hierarchical modeling.
- **Impact on ML**: Guides order of loss term weighting in multi-label neural networks.
- **Recommended Action**: Prioritize error analysis on rare tail labels (`threat`, `identity_hate`, `severe_toxic`).

---

## 4. Deep-Dive Interpretations & Best Practices

### Business Interpretation
The dataset exhibits extreme positive class sparsity. While general toxicity (`toxic`) affects ~10% of comments, severe violations (`threat`, `identity_hate`) affect less than 1% of traffic.

### Technical Interpretation
Multi-label binary target distribution confirms severe class imbalance. Standard Binary Cross-Entropy (BCE) loss without positive class weighting will lead to underfitting on minority classes.

### Recommendations
1. **Loss Function Optimization**: Utilize **BCEWithLogitsLoss** with `pos_weight = (num_negatives / num_positives)` or **Focal Loss** ($\gamma = 2.0$).
2. **Evaluation Metrics**: Evaluate models strictly using **Macro F1**, **PR-AUC**, and **Hamming Loss**.
3. **Stratified Splitting**: Use `multilabel_train_test_split` (Iterative Stratification) to preserve positive label ratios across folds.

---

## 5. Industry Best Practices & Technical Foundations

### Why Target Distribution Analysis is Important
Target distribution analysis identifies class imbalance before model design. In multi-label NLP, ignoring imbalance leads to the **Accuracy Paradox**, where a model predicting zero for all labels achieves 95%+ accuracy while failing 100% of toxicity detection objectives.

### How Imbalance Affects Multi-Label Evaluation Metrics
- **Accuracy**: Flawed metric (a model predicting 0 for `threat` gets ~99.5% accuracy).
- **ROC-AUC**: Can be overly optimistic when negative class is huge.
- **PR-AUC**: Measures true precision and recall trade-offs on minority positive class.
- **Macro F1**: Unweighted average across all 6 labels, giving equal voice to rare classes (`threat`) and common classes (`toxic`).

### Interview Q&A

#### Q1: Why is Macro F1 preferred over Micro F1 for imbalanced multi-label toxicity classification?
**Answer**: Micro F1 pools all true positives, false positives, and false negatives globally, allowing frequent classes (`toxic`, `insult`) to dominate the metric. Macro F1 calculates the unweighted mean of F1 scores across each label individually, ensuring rare but high-risk categories (`threat`, `identity_hate`) are weighted equally in performance evaluation.

#### Q2: How does Focal Loss address extreme class imbalance in multi-label neural networks?
**Answer**: Focal Loss adds a modulating factor $(1 - p_t)^\gamma$ to standard BCE loss. When a sample is easy and correctly classified ($p_t \to 1$), the modulating factor goes to 0, down-weighting well-classified negative examples and forcing model gradients to focus on hard, rare positive samples.
"""

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_md)

    logger.info(f"Target Label Distribution Analysis Report exported to {report_path}")

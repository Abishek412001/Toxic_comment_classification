"""
Configuration Manager Module for Evaluation.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from src.evaluation.constants import DEFAULT_THRESHOLD, DEFAULT_EVALUATION_DIR, TARGET_LABELS


@dataclass
class EvaluationConfig:
    """Dataclass storing evaluation thresholds, metric options, and export settings."""

    default_threshold: float = DEFAULT_THRESHOLD
    per_label_thresholds: Optional[Dict[str, float]] = None
    target_labels: List[str] = field(default_factory=lambda: list(TARGET_LABELS))
    output_dir: str = DEFAULT_EVALUATION_DIR
    save_plots: bool = True
    save_reports: bool = True
    optimization_metric: str = "macro_f1"  # "macro_f1", "micro_f1", "roc_auc"

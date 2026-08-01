"""
Configuration Manager Module for Explainable AI.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from src.xai.constants import DEFAULT_TOP_K_FEATURES, DEFAULT_XAI_DIR


@dataclass
class XAIConfig:
    """Dataclass storing XAI explainer settings, sample counts, and feature bounds."""

    method: str = "shap"  # "shap", "lime", "hybrid"
    num_features: int = DEFAULT_TOP_K_FEATURES
    num_samples: int = 100
    batch_size: int = 32
    device: str = "cpu"
    output_dir: str = DEFAULT_XAI_DIR
    save_plots: bool = True
    save_reports: bool = True

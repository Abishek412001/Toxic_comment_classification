"""
Configuration Manager Module for Sentiment Analysis.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from src.sentiment.constants import (
    SENTIMENT_LABELS,
    DEFAULT_POSITIVE_THRESHOLD,
    DEFAULT_NEGATIVE_THRESHOLD,
    DEFAULT_SENTIMENT_DIR,
)


@dataclass
class SentimentConfig:
    """Dataclass storing engine settings, thresholds, and execution options."""

    engine_type: str = "vader"  # "vader", "textblob", "transformer", "hybrid"
    model_name: str = "distilbert-base-uncased-finetuned-sst-2-english"
    pos_threshold: float = DEFAULT_POSITIVE_THRESHOLD
    neg_threshold: float = DEFAULT_NEGATIVE_THRESHOLD
    batch_size: int = 32
    device: str = "cpu"
    language: str = "english"
    output_dir: str = DEFAULT_SENTIMENT_DIR
    save_plots: bool = True
    save_reports: bool = True

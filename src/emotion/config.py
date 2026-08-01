"""
Configuration Manager Module for Emotion Mining.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from src.emotion.constants import EMOTION_LABELS, DEFAULT_EMOTION_DIR


@dataclass
class EmotionConfig:
    """Dataclass storing emotion engine settings, top-k parameters, and device options."""

    engine_type: str = "nrc"  # "nrc", "transformer", "hybrid"
    model_name: str = "j-hartmann/emotion-english-distilroberta-base"
    top_k: int = 3
    batch_size: int = 32
    device: str = "cpu"
    language: str = "english"
    output_dir: str = DEFAULT_EMOTION_DIR
    save_plots: bool = True
    save_reports: bool = True

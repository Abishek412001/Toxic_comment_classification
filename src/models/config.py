"""
Configuration Manager Module for Model Development.
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any
from src.models.constants import DEFAULT_RANDOM_STATE, DEFAULT_MODEL_DIR, DEFAULT_THRESHOLD


@dataclass
class ModelConfig:
    """Dataclass storing model hyperparameters and execution options."""

    model_name: str = "logistic_regression"
    random_state: int = DEFAULT_RANDOM_STATE
    batch_size: int = 64
    learning_rate: float = 0.001
    max_epochs: int = 10
    early_stopping_patience: int = 3
    threshold: float = DEFAULT_THRESHOLD
    device: str = "cpu"  # "cpu" or "cuda"
    save_model: bool = True
    save_tokenizer: bool = True
    use_gpu: bool = False
    artifact_dir: str = DEFAULT_MODEL_DIR
    extra_params: Optional[Dict[str, Any]] = None

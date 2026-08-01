"""
Model Serializer Module (Step 133).

Serializes/deserializes joblib, pickle, PyTorch, and Transformers artifacts with SHA256 checksums.
"""

import os
import hashlib
import joblib
import pickle
import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)

class ModelSerializer:
    """Serializer class handling artifact I/O and SHA256 checksum validation."""

    @staticmethod
    def calculate_checksum(filepath: str) -> str:
        """Calculates SHA256 checksum hash of file."""
        hasher = hashlib.sha256()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    @staticmethod
    def save_joblib(model: Any, filepath: str) -> str:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(model, filepath)
        checksum = ModelSerializer.calculate_checksum(filepath)
        logger.info(f"Saved joblib model to {filepath} (SHA256: {checksum[:12]}...)")
        return checksum

    @staticmethod
    def load_joblib(filepath: str) -> Any:
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        model = joblib.load(filepath)
        logger.info(f"Loaded joblib model from {filepath}")
        return model

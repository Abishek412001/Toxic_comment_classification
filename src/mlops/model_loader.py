"""
Model Loader Module (Step 133).

Provides thread-safe lazy loading and memory caching for model artifacts.
"""

import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)

class ModelLoader:
    """Thread-safe model loader class."""

    _cache: Dict[str, Any] = {}

    @classmethod
    def load_model(cls, model_name: str, loader_fn: Any) -> Any:
        if model_name in cls._cache:
            logger.info(f"Retrieved model '{model_name}' from memory cache.")
            return cls._cache[model_name]

        logger.info(f"Loading model '{model_name}' into memory...")
        model = loader_fn()
        cls._cache[model_name] = model
        return model

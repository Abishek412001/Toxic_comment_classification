"""
Model Registry Module.

Manages versioned model artifacts, metadata, and production promotion.
"""

import os
import json
import logging
from typing import Dict, Any, Optional
from src.models.constants import DEFAULT_REGISTRY_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class ModelRegistry:
    """Registry managing model artifact versions and production promotions."""

    def __init__(self, registry_dir: str = DEFAULT_REGISTRY_DIR):
        """Initializes model registry directory.

        Args:
            registry_dir: Output registry directory.
        """
        self.registry_dir = registry_dir
        os.makedirs(self.registry_dir, exist_ok=True)
        self.registry_file = os.path.join(self.registry_dir, "registry.json")
        self._load_registry()

    def _load_registry(self) -> None:
        if os.path.exists(self.registry_file):
            with open(self.registry_file, "r", encoding="utf-8") as f:
                self.records = json.load(f)
        else:
            self.records = {}

    def _save_registry(self) -> None:
        with open(self.registry_file, "w", encoding="utf-8") as f:
            json.dump(self.records, f, indent=2)

    def register_model(self, model_name: str, version: str, artifact_path: str, metrics: Dict[str, Any]) -> None:
        """Registers a versioned model artifact.

        Args:
            model_name: Model identifier.
            version: Version string (e.g. "v1.0.0").
            artifact_path: Filepath to saved model artifact.
            metrics: Evaluation metrics dictionary.
        """
        key = f"{model_name}:{version}"
        self.records[key] = {
            "model_name": model_name,
            "version": version,
            "artifact_path": artifact_path,
            "metrics": metrics,
            "stage": "Staging",
        }
        self._save_registry()
        logger.info(f"Registered model version {key} in Staging.")

    def promote_to_production(self, model_name: str, version: str) -> None:
        """Promotes a registered model version to Production stage.

        Args:
            model_name: Model identifier.
            version: Version string.
        """
        target_key = f"{model_name}:{version}"
        for key, record in self.records.items():
            if record["model_name"] == model_name:
                record["stage"] = "Archived"
        if target_key in self.records:
            self.records[target_key]["stage"] = "Production"
            self._save_registry()
            logger.info(f"Promoted model {target_key} to Production stage.")

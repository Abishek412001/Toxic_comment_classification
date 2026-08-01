"""
Model Registry Module (Step 133).

Manages model versioning, metadata tags, and active production aliases.
"""

import os
import json
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class ModelRegistry:
    """Registry class managing model versions and metadata manifests."""

    def __init__(self, manifest_path: str = "models/registry.json"):
        self.manifest_path = manifest_path
        self.registry = self._load_manifest()

    def _load_manifest(self) -> Dict[str, Any]:
        if os.path.exists(self.manifest_path):
            with open(self.manifest_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {"production_champion": "distilbert_multilabel_v1", "models": {}}

    def register_model(self, model_id: str, version: str, metadata: Dict[str, Any]) -> None:
        self.registry["models"][model_id] = {
            "version": version,
            "metadata": metadata,
            "status": "registered",
        }
        os.makedirs(os.path.dirname(self.manifest_path), exist_ok=True)
        with open(self.manifest_path, "w", encoding="utf-8") as f:
            json.dump(self.registry, f, indent=2)
        logger.info(f"Registered model '{model_id}' (version: {version})")

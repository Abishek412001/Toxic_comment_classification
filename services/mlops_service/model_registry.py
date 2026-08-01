"""
Enterprise Model Registry & Version Governance Engine.
"""

import uuid
from typing import Dict, List, Optional, Any
from datetime import datetime
from services.mlops_service.schemas import (
    ModelRegistrationRequest,
    ModelRegistrationResponse,
    ModelPromotionRequest,
    ModelStageEnum,
)
from opentrust_core.exceptions import NotFoundError, ValidationError

MODEL_REGISTRY_STORE: Dict[str, List[Dict[str, Any]]] = {
    "toxicity_classifier": [
        {
            "model_id": "mdl_v1_0_0",
            "model_name": "toxicity_classifier",
            "version": "1.0.0",
            "stage": ModelStageEnum.PRODUCTION.value,
            "metrics": {"f1_score": 0.942, "roc_auc": 0.985},
            "registered_at": datetime.utcnow(),
        }
    ]
}


class ModelRegistry:
    """Enterprise Model Registry & Promotion Engine."""

    def register_model(self, request: ModelRegistrationRequest) -> ModelRegistrationResponse:
        """Registers a new model version into the registry."""
        model_id = f"mdl_{uuid.uuid4().hex[:8]}"

        model_entry = {
            "model_id": model_id,
            "model_name": request.model_name,
            "version": request.version,
            "stage": ModelStageEnum.DEVELOPMENT.value,
            "metrics": request.metrics,
            "registered_at": datetime.utcnow(),
        }

        if request.model_name not in MODEL_REGISTRY_STORE:
            MODEL_REGISTRY_STORE[request.model_name] = []

        MODEL_REGISTRY_STORE[request.model_name].append(model_entry)

        return ModelRegistrationResponse(
            model_id=model_id,
            model_name=request.model_name,
            version=request.version,
            stage=ModelStageEnum.DEVELOPMENT,
        )

    def promote_model(self, request: ModelPromotionRequest) -> ModelRegistrationResponse:
        """Promotes a model version to Production/Staging and archives previous version."""
        versions = MODEL_REGISTRY_STORE.get(request.model_name, [])
        target = next((v for v in versions if v["version"] == request.version), None)

        if not target:
            raise NotFoundError(f"Model '{request.model_name}' version '{request.version}' not found.")

        # Demote current production version to Archived if target is Production
        if request.target_stage == ModelStageEnum.PRODUCTION:
            for v in versions:
                if v["stage"] == ModelStageEnum.PRODUCTION.value:
                    v["stage"] = ModelStageEnum.ARCHIVED.value

        target["stage"] = request.target_stage.value

        return ModelRegistrationResponse(
            model_id=target["model_id"],
            model_name=target["model_name"],
            version=target["version"],
            stage=request.target_stage,
        )

    def rollback_model(self, model_name: str, rollback_to_version: Optional[str] = None) -> ModelRegistrationResponse:
        """Instantly rolls back Production to previous version."""
        versions = MODEL_REGISTRY_STORE.get(model_name, [])
        if not versions:
            raise NotFoundError(f"No versions registered for model '{model_name}'.")

        # Find candidate version to promote
        if rollback_to_version:
            target = next((v for v in versions if v["version"] == rollback_to_version), None)
        else:
            # Pick latest archived / non-production version
            target = next((v for v in reversed(versions) if v["stage"] != ModelStageEnum.PRODUCTION.value), None)

        if not target:
            raise ValidationError(f"No valid rollback candidate version found for model '{model_name}'.")

        return self.promote_model(
            ModelPromotionRequest(
                model_name=model_name,
                version=target["version"],
                target_stage=ModelStageEnum.PRODUCTION,
            )
        )


model_registry = ModelRegistry()

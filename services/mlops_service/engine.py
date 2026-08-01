"""
Enterprise MLOps Orchestrator Engine combining Registry, Drift, Retraining & Observability.
"""

import uuid
from services.mlops_service.model_registry import model_registry
from services.mlops_service.drift_detector import drift_detector
from services.mlops_service.observability import observability_manager
from services.mlops_service.schemas import (
    ModelRegistrationRequest,
    ModelRegistrationResponse,
    ModelPromotionRequest,
    ModelRollbackRequest,
    DriftDetectionRequest,
    DriftDetectionResponse,
    RetrainingTriggerRequest,
    RetrainingTriggerResponse,
    ObservabilityMetricsResponse,
)


class MLOpsEngine:
    """Enterprise MLOps Orchestrator Engine."""

    def register_model(self, request: ModelRegistrationRequest) -> ModelRegistrationResponse:
        return model_registry.register_model(request)

    def promote_model(self, request: ModelPromotionRequest) -> ModelRegistrationResponse:
        return model_registry.promote_model(request)

    def rollback_model(self, request: ModelRollbackRequest) -> ModelRegistrationResponse:
        return model_registry.rollback_model(request.model_name, request.rollback_to_version)

    def detect_drift(self, request: DriftDetectionRequest) -> DriftDetectionResponse:
        return drift_detector.detect_drift(request)

    def trigger_retraining(self, request: RetrainingTriggerRequest) -> RetrainingTriggerResponse:
        job_id = f"job_{uuid.uuid4().hex[:8]}"
        return RetrainingTriggerResponse(
            job_id=job_id,
            model_name=request.model_name,
            status="QUEUED",
        )

    def get_observability_metrics(self) -> ObservabilityMetricsResponse:
        return observability_manager.get_metrics()


mlops_engine = MLOpsEngine()

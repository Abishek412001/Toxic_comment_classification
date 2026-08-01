"""
Pydantic v2 Schemas for MLOps Registry, Drift Detection, Retraining, and Observability.
"""

from enum import Enum
from typing import List, Dict, Optional, Any
from datetime import datetime
from pydantic import Field
from opentrust_core.schemas.base import BaseSchema


class ModelStageEnum(str, Enum):
    DEVELOPMENT = "Development"
    STAGING = "Staging"
    PRODUCTION = "Production"
    ARCHIVED = "Archived"


class ModelRegistrationRequest(BaseSchema):
    model_name: str = Field(min_length=1)
    version: str = Field(pattern=r"^\d+\.\d+\.\d+$")
    framework: str = Field(default="PyTorch / Transformers")
    metrics: Dict[str, float] = Field(default_factory=dict)
    artifact_path: str = Field(default="models/distilbert_v1.pt")


class ModelRegistrationResponse(BaseSchema):
    model_id: str
    model_name: str
    version: str
    stage: ModelStageEnum
    registered_at: datetime = Field(default_factory=datetime.utcnow)


class ModelPromotionRequest(BaseSchema):
    model_name: str
    version: str
    target_stage: ModelStageEnum = ModelStageEnum.PRODUCTION


class ModelRollbackRequest(BaseSchema):
    model_name: str
    rollback_to_version: Optional[str] = None


class DriftDetectionRequest(BaseSchema):
    model_name: str = "toxicity_classifier"
    baseline_sample_size: int = 1000
    current_sample_size: int = 1000


class DriftDetectionResponse(BaseSchema):
    model_name: str
    drift_detected: bool
    psi_score: float = Field(ge=0.0, description="Population Stability Index score")
    kl_divergence: float = Field(ge=0.0)
    js_divergence: float = Field(ge=0.0)
    recommendation: str  # NO_ACTION, MONITOR, RETRAIN_RECOMMENDED
    checked_at: datetime = Field(default_factory=datetime.utcnow)


class RetrainingTriggerRequest(BaseSchema):
    model_name: str
    reason: str = "Drift detected or scheduled maintenance"


class RetrainingTriggerResponse(BaseSchema):
    job_id: str
    model_name: str
    status: str  # QUEUED, IN_PROGRESS, COMPLETED
    triggered_at: datetime = Field(default_factory=datetime.utcnow)


class ObservabilityMetricsResponse(BaseSchema):
    active_models_in_production: int
    avg_latency_ms: float
    requests_per_minute: int
    error_rate_percent: float
    gpu_utilization_percent: float
    cpu_utilization_percent: float
    estimated_hourly_cost_usd: float

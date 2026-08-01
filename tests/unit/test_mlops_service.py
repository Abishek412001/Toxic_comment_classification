"""
Unit Tests for MLOps Engine, Model Registry, Drift Detection & Observability.
"""

import unittest
from services.mlops_service.engine import MLOpsEngine
from services.mlops_service.schemas import (
    ModelRegistrationRequest,
    ModelPromotionRequest,
    ModelRollbackRequest,
    DriftDetectionRequest,
    ModelStageEnum,
)


class TestMLOpsService(unittest.TestCase):
    def setUp(self):
        self.engine = MLOpsEngine()

    def test_model_registration_and_promotion(self):
        reg_req = ModelRegistrationRequest(
            model_name="toxicity_classifier",
            version="2.0.0",
            metrics={"f1_score": 0.965},
        )
        reg_res = self.engine.register_model(reg_req)
        self.assertEqual(reg_res.version, "2.0.0")
        self.assertEqual(reg_res.stage, ModelStageEnum.DEVELOPMENT)

        promo_req = ModelPromotionRequest(
            model_name="toxicity_classifier",
            version="2.0.0",
            target_stage=ModelStageEnum.PRODUCTION,
        )
        promo_res = self.engine.promote_model(promo_req)
        self.assertEqual(promo_res.stage, ModelStageEnum.PRODUCTION)

    def test_model_rollback(self):
        rollback_req = ModelRollbackRequest(model_name="toxicity_classifier")
        res = self.engine.rollback_model(rollback_req)
        self.assertIsNotNone(res.version)

    def test_drift_detection(self):
        drift_req = DriftDetectionRequest(model_name="toxicity_classifier")
        res = self.engine.detect_drift(drift_req)
        self.assertGreaterEqual(res.psi_score, 0.0)
        self.assertIsNotNone(res.recommendation)

    def test_observability_metrics(self):
        res = self.engine.get_observability_metrics()
        self.assertGreater(res.active_models_in_production, 0)
        self.assertGreater(res.requests_per_minute, 0)


if __name__ == "__main__":
    unittest.main()

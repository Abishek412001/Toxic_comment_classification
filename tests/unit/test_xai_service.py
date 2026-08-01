"""
Unit Tests for XAI Explainer Engine (SHAP & LIME).
"""

import unittest
from services.xai_service.engine import XAIEngine
from services.xai_service.schemas import XAIRequest, BatchXAIRequest, ExplainerMethodEnum


class TestXAIEngine(unittest.TestCase):
    def setUp(self):
        self.engine = XAIEngine()

    def test_shap_explanation(self):
        req = XAIRequest(text="This product has a stupid bug and I hate it.", method=ExplainerMethodEnum.SHAP)
        res = self.engine.explain_single(req)
        self.assertEqual(res.explainer_method, ExplainerMethodEnum.SHAP)
        self.assertIn("FLAGGED", res.prediction)
        self.assertGreater(len(res.feature_contributions), 0)

    def test_lime_explanation(self):
        req = XAIRequest(text="Clean benign text with no issues.", method=ExplainerMethodEnum.LIME)
        res = self.engine.explain_single(req)
        self.assertEqual(res.explainer_method, ExplainerMethodEnum.LIME)
        self.assertIn("PASS", res.prediction)

    def test_batch_explanation(self):
        req = BatchXAIRequest(texts=["First text", "Second text with hate"])
        res = self.engine.explain_batch(req)
        self.assertEqual(res.total_processed, 2)


if __name__ == "__main__":
    unittest.main()

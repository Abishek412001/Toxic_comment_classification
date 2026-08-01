"""
Unit Tests for Decision Intelligence Engine & Audit Trail.
"""

import unittest
from services.decision_service.engine import DecisionEngine
from services.decision_service.schemas import DecisionEvaluationRequest


class TestDecisionEngine(unittest.TestCase):
    def setUp(self):
        self.engine = DecisionEngine()

    def test_low_confidence_hitl_trigger(self):
        req = DecisionEvaluationRequest(
            text="Ambiguous text snippet",
            risk_score=0.40,
            confidence_score=0.55,
            moderation_action="PASS",
        )
        res = self.engine.evaluate_decision(req)
        self.assertTrue(res.trigger_hitl_review)
        self.assertIn("Low model prediction confidence", res.hitl_reason)

    def test_high_risk_hitl_trigger(self):
        req = DecisionEvaluationRequest(
            text="Critical threat text",
            risk_score=0.90,
            confidence_score=0.95,
            moderation_action="FLAG",
        )
        res = self.engine.evaluate_decision(req)
        self.assertTrue(res.trigger_hitl_review)
        self.assertEqual(res.risk_level, "CRITICAL")

    def test_audit_trail_logging(self):
        req = DecisionEvaluationRequest(
            text="Audit log test text",
            risk_score=0.20,
            confidence_score=0.95,
            moderation_action="PASS",
        )
        self.engine.evaluate_decision(req)
        trail = self.engine.get_audit_trail()
        self.assertGreater(len(trail), 0)


if __name__ == "__main__":
    unittest.main()

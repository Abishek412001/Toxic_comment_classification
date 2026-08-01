"""
Unit Tests for Moderation Classifier Engine and Policy Decision Engine.
"""

import unittest
from services.moderation_service.engine import ModerationEngine
from services.moderation_service.schemas import ModerationRequest, BatchModerationRequest, ActionEnum


class TestModerationEngine(unittest.TestCase):
    def setUp(self):
        self.engine = ModerationEngine()

    def test_benign_text_predict(self):
        req = ModerationRequest(text="This is a great product and awesome community.")
        res = self.engine.predict_single(req)

        self.assertFalse(res.flagged)
        self.assertEqual(res.action, ActionEnum.PASS)
        self.assertLess(res.overall_risk_score, 0.5)
        self.assertGreater(res.latency_ms, 0.0)

    def test_toxic_text_predict(self):
        req = ModerationRequest(text="I hate you, you are a stupid idiot!")
        res = self.engine.predict_single(req)

        self.assertTrue(res.flagged)
        self.assertIn(res.action, [ActionEnum.FLAG, ActionEnum.BLOCK])
        self.assertGreater(res.categories.toxic, 0.5)

    def test_batch_prediction(self):
        req = BatchModerationRequest(
            texts=[
                "Hello, hope you have a nice day!",
                "I am going to kill you stupid idiot!",
                "Normal conversation post.",
            ]
        )
        res = self.engine.predict_batch(req)

        self.assertEqual(res.total_processed, 3)
        self.assertEqual(res.flagged_count, 1)
        self.assertEqual(len(res.results), 3)


if __name__ == "__main__":
    unittest.main()

"""
Unit Tests for Sentiment Analysis Engine across VADER, TextBlob, and Ensemble.
"""

import unittest
from services.sentiment_service.engine import SentimentEngine
from services.sentiment_service.schemas import (
    SentimentRequest,
    BatchSentimentRequest,
    SentimentLabelEnum,
    EngineTypeEnum,
)


class TestSentimentEngine(unittest.TestCase):
    def setUp(self):
        self.engine = SentimentEngine()

    def test_vader_positive(self):
        req = SentimentRequest(text="This product is absolutely wonderful and amazing!", engine=EngineTypeEnum.VADER)
        res = self.engine.analyze_single(req)
        self.assertEqual(res.label, SentimentLabelEnum.POSITIVE)
        self.assertGreater(res.compound_score, 0.5)

    def test_textblob_negative(self):
        req = SentimentRequest(text="This experience was terrible and awful.", engine=EngineTypeEnum.TEXTBLOB)
        res = self.engine.analyze_single(req)
        self.assertEqual(res.label, SentimentLabelEnum.NEGATIVE)
        self.assertLess(res.polarity, -0.2)

    def test_ensemble_batch(self):
        req = BatchSentimentRequest(
            texts=[
                "I love this awesome service!",
                "The meeting is at 2 PM today.",
                "I hate this terrible bug.",
            ],
            engine=EngineTypeEnum.ENSEMBLE,
        )
        res = self.engine.analyze_batch(req)
        self.assertEqual(res.total_processed, 3)
        self.assertEqual(res.positive_count, 1)
        self.assertEqual(res.negative_count, 1)
        self.assertEqual(res.neutral_count, 1)


if __name__ == "__main__":
    unittest.main()

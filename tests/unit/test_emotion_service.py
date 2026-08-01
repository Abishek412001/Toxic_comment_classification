"""
Unit Tests for Multi-Class Emotion Detection Engine.
"""

import unittest
from services.emotion_service.engine import EmotionEngine
from services.emotion_service.schemas import EmotionRequest, BatchEmotionRequest


class TestEmotionEngine(unittest.TestCase):
    def setUp(self):
        self.engine = EmotionEngine()

    def test_joy_detection(self):
        req = EmotionRequest(text="I am so happy, excited, and full of joy!", top_n=3)
        res = self.engine.detect_single(req)
        self.assertEqual(res.dominant_emotion, "joy")
        self.assertEqual(len(res.top_emotions), 3)

    def test_fear_detection(self):
        req = EmotionRequest(text="I am scared, afraid, and terrified of danger.", top_n=2)
        res = self.engine.detect_single(req)
        self.assertEqual(res.dominant_emotion, "fear")

    def test_batch_detection(self):
        req = BatchEmotionRequest(
            texts=[
                "I am happy and excited!",
                "I am furious and mad!",
            ]
        )
        res = self.engine.detect_batch(req)
        self.assertEqual(res.total_processed, 2)
        self.assertEqual(res.emotion_counts.get("joy", 0), 1)


if __name__ == "__main__":
    unittest.main()

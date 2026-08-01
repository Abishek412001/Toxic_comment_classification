import unittest
from src.emotion.emotion_pipeline import EmotionPipeline
from src.emotion.config import EmotionConfig

class TestEmotionPipeline(unittest.TestCase):
    def test_pipeline_execution(self):
        config = EmotionConfig(engine_type="nrc")
        pipeline = EmotionPipeline(config=config)
        res = pipeline.analyze_text("I am very happy and delighted!")
        self.assertIn("emotion_label", res)
        self.assertIn("top_emotions", res)
        self.assertEqual(res["emotion_label"], "joy")

if __name__ == "__main__":
    unittest.main()

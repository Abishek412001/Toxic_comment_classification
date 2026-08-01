import unittest
from src.sentiment.sentiment_pipeline import SentimentPipeline
from src.sentiment.config import SentimentConfig

class TestSentimentPipeline(unittest.TestCase):
    def test_pipeline_execution(self):
        config = SentimentConfig(engine_type="vader")
        pipeline = SentimentPipeline(config=config)
        res = pipeline.analyze_text("This is an awesome and positive comment!")
        self.assertIn("sentiment_label", res)
        self.assertEqual(res["sentiment_label"], "positive")

if __name__ == "__main__":
    unittest.main()

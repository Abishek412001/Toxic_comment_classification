import unittest
from src.sentiment.production_sentiment_pipeline import ProductionSentimentPipeline

class TestProductionSentiment(unittest.TestCase):
    def test_production_pipeline(self):
        pipeline = ProductionSentimentPipeline()
        res = pipeline.predict_single("Awesome work!")
        self.assertEqual(res["sentiment_label"], "positive")
        rest_res = pipeline.format_rest_payload(res)
        self.assertEqual(rest_res["status"], "success")

if __name__ == "__main__":
    unittest.main()

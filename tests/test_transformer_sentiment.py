import unittest
from src.sentiment.transformer_analyzer import TransformerSentimentAnalyzer

class TestTransformerSentiment(unittest.TestCase):
    def test_transformer_analysis(self):
        analyzer = TransformerSentimentAnalyzer()
        res = analyzer.analyze("Great product and amazing quality!")
        self.assertEqual(res["sentiment_label"], "positive")

if __name__ == "__main__":
    unittest.main()

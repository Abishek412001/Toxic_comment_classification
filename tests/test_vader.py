import unittest
from src.sentiment.vader_analyzer import VADERAnalyzer

class TestVADERAnalyzer(unittest.TestCase):
    def test_vader_analysis(self):
        analyzer = VADERAnalyzer()
        res = analyzer.analyze("I love this wonderful system!")
        self.assertEqual(res["sentiment_label"], "positive")
        self.assertGreater(res["compound_score"], 0.0)

if __name__ == "__main__":
    unittest.main()

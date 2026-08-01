import unittest
from src.sentiment.textblob_analyzer import TextBlobAnalyzer

class TestTextBlobAnalyzer(unittest.TestCase):
    def test_textblob_analysis(self):
        analyzer = TextBlobAnalyzer()
        res = analyzer.analyze("This is a terrible and bad experience.")
        self.assertEqual(res["sentiment_label"], "negative")
        self.assertLess(res["polarity"], 0.0)

if __name__ == "__main__":
    unittest.main()

import unittest
from src.emotion.nrc_analyzer import NRCEmotionAnalyzer

class TestNRCEmotionAnalyzer(unittest.TestCase):
    def test_nrc_analysis(self):
        analyzer = NRCEmotionAnalyzer()
        res = analyzer.analyze("I feel mad, furious, and full of rage!")
        self.assertEqual(res["emotion_label"], "anger")
        self.assertGreater(res["confidence_score"], 0.0)

if __name__ == "__main__":
    unittest.main()

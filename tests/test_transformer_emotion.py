import unittest
from src.emotion.transformer_analyzer import TransformerEmotionAnalyzer

class TestTransformerEmotion(unittest.TestCase):
    def test_transformer_analysis(self):
        analyzer = TransformerEmotionAnalyzer()
        res = analyzer.analyze("I am super excited and happy!")
        self.assertEqual(res["emotion_label"], "joy")

if __name__ == "__main__":
    unittest.main()

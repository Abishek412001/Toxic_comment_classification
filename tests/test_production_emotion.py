import unittest
from src.emotion.production_emotion_pipeline import ProductionEmotionPipeline

class TestProductionEmotion(unittest.TestCase):
    def test_production_pipeline(self):
        pipeline = ProductionEmotionPipeline()
        res = pipeline.predict_single("I am so happy!")
        self.assertEqual(res["emotion_label"], "joy")
        rest_res = pipeline.format_rest_payload(res)
        self.assertEqual(rest_res["status"], "success")

if __name__ == "__main__":
    unittest.main()

import unittest
from src.xai.xai_pipeline import XAIPipeline
from src.xai.config import XAIConfig

class MockModel:
    def predict(self, X):
        return [1]

class TestXAIPipeline(unittest.TestCase):
    def test_pipeline_execution(self):
        config = XAIConfig(method="shap")
        pipeline = XAIPipeline(config=config)
        res = pipeline.explain_text("This is a bad and stupid comment", MockModel())
        self.assertIn("positive_contributors", res)
        self.assertIn("negative_contributors", res)

if __name__ == "__main__":
    unittest.main()

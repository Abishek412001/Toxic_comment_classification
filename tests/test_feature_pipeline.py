import unittest
import numpy as np
from src.features.feature_pipeline import FeaturePipeline
from src.features.feature_factory import FeatureFactory
from src.features.base_feature_extractor import BaseFeatureExtractor
from src.features.config import FeatureConfig

class MockPipelineExtractor(BaseFeatureExtractor):
    def __init__(self, config=None):
        super().__init__(name="MockPipelineExtractor")
        self.config = config

    def fit(self, texts):
        self.is_fitted = True
        return self

    def transform(self, texts):
        return np.zeros((len(texts), 4))

    def save(self, filepath):
        pass

    def load(self, filepath):
        self.is_fitted = True
        return self

    def get_feature_names(self):
        return ["f1", "f2", "f3", "f4"]

class TestFeaturePipeline(unittest.TestCase):
    def setUp(self):
        FeatureFactory.register("mock_pipeline", MockPipelineExtractor)

    def test_pipeline_fit_transform(self):
        config = FeatureConfig(feature_type="mock_pipeline")
        pipeline = FeaturePipeline(config=config)
        matrix = pipeline.fit_transform(["comment 1", "comment 2"])
        self.assertEqual(matrix.shape, (2, 4))

    def test_pipeline_custom_extractor_injection(self):
        custom_extractor = MockPipelineExtractor()
        pipeline = FeaturePipeline(extractor=custom_extractor)
        matrix = pipeline.fit_transform(["test comment"])
        self.assertEqual(matrix.shape, (1, 4))

if __name__ == "__main__":
    unittest.main()

import unittest
from src.features.feature_factory import FeatureFactory
from src.features.base_feature_extractor import BaseFeatureExtractor
from src.features.config import FeatureConfig
from src.features.exceptions import ConfigurationError

class MockExtractor(BaseFeatureExtractor):
    def __init__(self, config=None):
        super().__init__(name="MockExtractor")
        self.config = config

    def fit(self, texts):
        self.is_fitted = True
        return self

    def transform(self, texts):
        return [[1.0, 2.0] for _ in texts]

    def save(self, filepath):
        pass

    def load(self, filepath):
        return self

    def get_feature_names(self):
        return ["f1", "f2"]

class TestFeatureFactory(unittest.TestCase):
    def setUp(self):
        FeatureFactory.register("mock", MockExtractor)

    def test_factory_registration_and_creation(self):
        config = FeatureConfig(feature_type="mock")
        extractor = FeatureFactory.create(config)
        self.assertIsInstance(extractor, MockExtractor)
        self.assertEqual(extractor.name, "MockExtractor")

    def test_unknown_feature_type(self):
        config = FeatureConfig(feature_type="unknown_type")
        with self.assertRaises(ConfigurationError):
            FeatureFactory.create(config)

if __name__ == "__main__":
    unittest.main()

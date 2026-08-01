import unittest
import numpy as np
from src.models.model_factory import ModelFactory
from src.models.config import ModelConfig
from src.models.logistic_regression import MultiLabelLogisticRegression

class TestModelPipeline(unittest.TestCase):
    def test_factory_registration(self):
        config = ModelConfig(model_name="logistic_regression")
        model = ModelFactory.create(config)
        self.assertIsInstance(model, MultiLabelLogisticRegression)

if __name__ == "__main__":
    unittest.main()

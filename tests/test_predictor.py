import unittest
import numpy as np
from src.models.logistic_regression import MultiLabelLogisticRegression
from src.models.predictor import ModelPredictor

class TestModelPredictor(unittest.TestCase):
    def test_prediction(self):
        model = MultiLabelLogisticRegression()
        X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
        y = np.array([[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]])
        model.fit(X, y)
        predictor = ModelPredictor(model)
        batch_res = predictor.predict_batch(X)
        self.assertEqual(batch_res["predictions"].shape, (4, 6))

if __name__ == "__main__":
    unittest.main()

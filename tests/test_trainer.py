import unittest
import numpy as np
from src.models.logistic_regression import MultiLabelLogisticRegression
from src.models.trainer import ModelTrainer

class TestModelTrainer(unittest.TestCase):
    def test_training(self):
        model = MultiLabelLogisticRegression()
        trainer = ModelTrainer(model)
        X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
        y = np.array([[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]])
        res = trainer.train(X, y)
        self.assertEqual(res["status"], "success")
        self.assertTrue(model.is_fitted)

if __name__ == "__main__":
    unittest.main()

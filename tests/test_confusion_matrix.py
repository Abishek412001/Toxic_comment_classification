import unittest
import numpy as np
from src.evaluation.confusion_matrix_analysis import ConfusionMatrixAnalyzer

class TestConfusionMatrix(unittest.TestCase):
    def test_matrix_computation(self):
        y_true = np.ones((4, 6))
        y_pred = np.ones((4, 6))
        cm_data = ConfusionMatrixAnalyzer.compute_matrices(y_true, y_pred)
        self.assertIn("toxic", cm_data)
        self.assertEqual(cm_data["toxic"]["tp"], 4)

if __name__ == "__main__":
    unittest.main()

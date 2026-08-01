import unittest
import numpy as np
from src.evaluation.roc_auc_analysis import ROCAUCAnalyzer

class TestROCAUC(unittest.TestCase):
    def test_roc_curves(self):
        y_true = np.array([[1, 0, 1, 0, 1, 0], [0, 1, 0, 1, 0, 1]])
        y_proba = np.array([[0.9, 0.1, 0.9, 0.1, 0.9, 0.1], [0.1, 0.9, 0.1, 0.9, 0.1, 0.9]])
        roc_data = ROCAUCAnalyzer.compute_roc_curves(y_true, y_proba)
        self.assertIn("micro", roc_data["auc"])
        self.assertEqual(roc_data["auc"]["micro"], 1.0)

if __name__ == "__main__":
    unittest.main()

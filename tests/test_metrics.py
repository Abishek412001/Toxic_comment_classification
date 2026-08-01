import unittest
import numpy as np
from src.models.metrics import compute_multilabel_metrics

class TestModelMetrics(unittest.TestCase):
    def test_metrics_computation(self):
        y_true = np.ones((2, 6))
        y_pred = np.ones((2, 6))
        y_proba = np.ones((2, 6)) * 0.9
        metrics = compute_multilabel_metrics(y_true, y_pred, y_proba)
        self.assertEqual(metrics["macro_f1"], 1.0)
        self.assertEqual(metrics["hamming_loss"], 0.0)

if __name__ == "__main__":
    unittest.main()

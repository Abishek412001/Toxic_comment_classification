import unittest
import numpy as np
from src.evaluation.threshold_optimizer import ThresholdOptimizer

class TestThresholdOptimizer(unittest.TestCase):
    def test_optimization(self):
        optimizer = ThresholdOptimizer()
        y_true = np.array([[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0]])
        y_proba = np.array([[0.8, 0.2, 0.1, 0.1, 0.1, 0.1], [0.2, 0.8, 0.1, 0.1, 0.1, 0.1]])
        thresholds = optimizer.optimize_per_label(y_true, y_proba)
        self.assertIn("toxic", thresholds)
        self.assertGreater(thresholds["toxic"], 0.0)

if __name__ == "__main__":
    unittest.main()

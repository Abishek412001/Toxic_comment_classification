import unittest
import numpy as np
from src.evaluation.one_vs_rest import OneVsRestEvaluator

class TestOneVsRest(unittest.TestCase):
    def test_evaluation(self):
        evaluator = OneVsRestEvaluator()
        y_true = np.ones((5, 6))
        y_proba = np.ones((5, 6)) * 0.8
        metrics = evaluator.evaluate(y_true, y_proba)
        self.assertEqual(metrics["macro_f1"], 1.0)

if __name__ == "__main__":
    unittest.main()

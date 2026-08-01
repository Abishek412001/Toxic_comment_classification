import unittest
import numpy as np
from src.evaluation.error_analysis import ErrorAnalyzer

class TestErrorAnalysis(unittest.TestCase):
    def test_error_analysis(self):
        y_true = np.array([[1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]])
        y_pred = np.array([[0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0]])
        texts = ["missed toxic comment", "false positive toxic comment"]
        err_res = ErrorAnalyzer.analyze_errors(y_true, y_pred, texts)
        self.assertEqual(err_res["total_errors"], 2)

if __name__ == "__main__":
    unittest.main()

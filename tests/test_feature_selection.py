import unittest
import numpy as np
from src.features.feature_selection import FeatureSelector

class TestFeatureSelector(unittest.TestCase):
    def test_chi2_selection(self):
        X = np.array([[1, 0, 5], [0, 2, 0], [1, 1, 4], [0, 0, 1]])
        y = np.array([1, 0, 1, 0])
        selector = FeatureSelector(method="chi2", k=2)
        X_sel = selector.fit_transform(X, y)
        self.assertEqual(X_sel.shape, (4, 2))

    def test_variance_selection(self):
        X = np.array([[1, 0, 5], [1, 2, 0], [1, 1, 4], [1, 0, 1]])
        y = np.array([1, 0, 1, 0])
        selector = FeatureSelector(method="variance", variance_threshold=0.01)
        X_sel = selector.fit_transform(X, y)
        self.assertLess(X_sel.shape[1], 3)

if __name__ == "__main__":
    unittest.main()

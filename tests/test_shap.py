import unittest
from src.xai.shap_explainer import SHAPExplainer

class MockModel:
    def predict(self, X):
        return [1]

class TestSHAPExplainer(unittest.TestCase):
    def test_shap_explanation(self):
        explainer = SHAPExplainer()
        res = explainer.explain_instance("You are an idiot and a fool", MockModel())
        self.assertEqual(res["method"], "shap")
        self.assertIn("positive_contributors", res)

if __name__ == "__main__":
    unittest.main()

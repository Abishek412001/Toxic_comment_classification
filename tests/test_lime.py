import unittest
from src.xai.lime_explainer import LIMEExplainer

class MockModel:
    def predict(self, X):
        return [1]

class TestLIMEExplainer(unittest.TestCase):
    def test_lime_explanation(self):
        explainer = LIMEExplainer()
        res = explainer.explain_instance("This is a horrible mistake", MockModel())
        self.assertEqual(res["method"], "lime")
        self.assertIn("positive_contributors", res)

if __name__ == "__main__":
    unittest.main()

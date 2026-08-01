import unittest
import numpy as np
from src.features.feature_validator import FeatureValidator
from src.features.exceptions import ValidationError

class TestFeatureValidator(unittest.TestCase):
    def test_valid_input_texts(self):
        texts = ["hello", "world"]
        res = FeatureValidator.validate_input_texts(texts)
        self.assertEqual(res, ["hello", "world"])

    def test_none_input_texts(self):
        with self.assertRaises(ValidationError):
            FeatureValidator.validate_input_texts(None)

    def test_empty_input_texts(self):
        with self.assertRaises(ValidationError):
            FeatureValidator.validate_input_texts([])

    def test_unfitted_state(self):
        with self.assertRaises(ValidationError):
            FeatureValidator.validate_fitted_state(False, "MockExtractor")

    def test_matrix_row_mismatch(self):
        matrix = np.zeros((5, 10))
        with self.assertRaises(ValidationError):
            FeatureValidator.validate_feature_matrix(matrix, expected_rows=10)

if __name__ == "__main__":
    unittest.main()

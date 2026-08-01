import unittest
from src.preprocessing.validator import TextValidator
from src.preprocessing.exceptions import InvalidInputError, EmptyTextError

class TestTextValidator(unittest.TestCase):
    def test_valid_text(self):
        res = TextValidator.validate_text("Hello World")
        self.assertEqual(res, "Hello World")

    def test_none_text(self):
        with self.assertRaises(InvalidInputError):
            TextValidator.validate_text(None)

    def test_non_string(self):
        with self.assertRaises(InvalidInputError):
            TextValidator.validate_text({"invalid": "dict"})

    def test_empty_string(self):
        with self.assertRaises(EmptyTextError):
            TextValidator.validate_text("   ", allow_empty=False)

if __name__ == "__main__":
    unittest.main()

import unittest
from src.preprocessing.lowercase import LowercaseTransformer, apply_lowercase, batch_lowercase

class TestLowercase(unittest.TestCase):
    def test_single_lowercase(self):
        self.assertEqual(apply_lowercase("HELLO WORLD 123!"), "hello world 123!")

    def test_unicode_preservation(self):
        self.assertEqual(apply_lowercase("RÉSUMÉ Café"), "résumé café")

    def test_batch_lowercase(self):
        res = batch_lowercase(["TEXT ONE", "TEXT TWO"])
        self.assertEqual(res, ["text one", "text two"])

if __name__ == "__main__":
    unittest.main()

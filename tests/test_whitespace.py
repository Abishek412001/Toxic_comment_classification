import unittest
from src.preprocessing.whitespace_normalizer import WhitespaceNormalizer

class TestWhitespaceNormalizer(unittest.TestCase):
    def test_whitespace_normalization(self):
        cleaner = WhitespaceNormalizer()
        res = cleaner.transform("  Hello \t\n  world \n\n ")
        self.assertEqual(res, "Hello world")

if __name__ == "__main__":
    unittest.main()

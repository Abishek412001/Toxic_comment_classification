import unittest
from src.preprocessing.punctuation_cleaner import PunctuationCleaner

class TestPunctuationCleaner(unittest.TestCase):
    def test_punctuation_removal(self):
        cleaner = PunctuationCleaner()
        res = cleaner.transform("Hello, world!!! How are you?")
        self.assertEqual(res, "Hello world How are you")

if __name__ == "__main__":
    unittest.main()

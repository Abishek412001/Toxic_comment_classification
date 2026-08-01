import unittest
from src.preprocessing.number_cleaner import NumberCleaner

class TestNumberCleaner(unittest.TestCase):
    def test_number_replacement(self):
        cleaner = NumberCleaner(replacement_token="0")
        res = cleaner.transform("I have 10 apples and 3.5 oranges")
        self.assertEqual(res, "I have 0 apples and 0 oranges")

    def test_word_embedded_digits(self):
        cleaner = NumberCleaner(replacement_token="0")
        res = cleaner.transform("Win32 app")
        self.assertEqual(res, "Win32 app")

if __name__ == "__main__":
    unittest.main()

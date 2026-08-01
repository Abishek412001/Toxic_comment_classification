import unittest
from src.preprocessing.special_character_cleaner import SpecialCharacterCleaner

class TestSpecialCharacterCleaner(unittest.TestCase):
    def test_special_char_removal(self):
        cleaner = SpecialCharacterCleaner()
        res = cleaner.transform("Symbol § © ™ test")
        self.assertEqual(res, "Symbol test")

if __name__ == "__main__":
    unittest.main()

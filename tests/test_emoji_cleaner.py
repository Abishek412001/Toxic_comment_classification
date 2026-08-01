import unittest
from src.preprocessing.emoji_cleaner import EmojiCleaner

class TestEmojiCleaner(unittest.TestCase):
    def test_emoji_demoji(self):
        cleaner = EmojiCleaner(demoji_to_text=True)
        res = cleaner.transform("I am angry 🤬")
        self.assertTrue(len(res) > 0)
        self.assertNotIn("🤬", res)

    def test_emoji_strip(self):
        cleaner = EmojiCleaner(demoji_to_text=False)
        res = cleaner.transform("Happy day 😊")
        self.assertEqual(res, "Happy day")

if __name__ == "__main__":
    unittest.main()

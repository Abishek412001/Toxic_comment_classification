import unittest
from src.preprocessing.url_cleaner import URLCleaner

class TestURLCleaner(unittest.TestCase):
    def test_url_removal(self):
        cleaner = URLCleaner(replacement_token="[URL]")
        res = cleaner.transform("Visit http://example.com/test for info")
        self.assertIn("[URL]", res)
        self.assertNotIn("http://example.com", res)

    def test_ftp_and_www(self):
        cleaner = URLCleaner(replacement_token="")
        res = cleaner.transform("Check www.google.com and ftp://files.org")
        self.assertEqual(res.strip(), "Check and")

if __name__ == "__main__":
    unittest.main()

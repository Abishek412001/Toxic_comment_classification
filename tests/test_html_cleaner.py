import unittest
from src.preprocessing.html_cleaner import HTMLCleaner

class TestHTMLCleaner(unittest.TestCase):
    def setUp(self):
        self.cleaner = HTMLCleaner()

    def test_standard_html(self):
        res = self.cleaner.transform("<p>This is <b>bold</b> text</p>")
        self.assertEqual(res, "This is bold text")

    def test_malformed_html(self):
        res = self.cleaner.transform("<div>Unclosed tag <span>content")
        self.assertNotIn("<div>", res)

if __name__ == "__main__":
    unittest.main()

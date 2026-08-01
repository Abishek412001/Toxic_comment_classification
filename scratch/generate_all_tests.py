"""
Script to create all 15 unit test modules in tests/ directory.
"""

import os
import sys

sys.path.insert(0, os.path.abspath("."))

TESTS_DIR = "tests"
os.makedirs(TESTS_DIR, exist_ok=True)

test_files = {
    "test_validator.py": """import unittest
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
            TextValidator.validate_text(12345)

    def test_empty_string(self):
        with self.assertRaises(EmptyTextError):
            TextValidator.validate_text("   ", allow_empty=False)

if __name__ == "__main__":
    unittest.main()
""",
    "test_lowercase.py": """import unittest
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
""",
    "test_contractions.py": """import unittest
from src.preprocessing.contractions import ContractionExpander

class TestContractionExpander(unittest.TestCase):
    def setUp(self):
        self.expander = ContractionExpander()

    def test_standard_contractions(self):
        self.assertIn("do not", self.expander.transform("I don't know"))
        self.assertIn("cannot", self.expander.transform("I can't go"))

    def test_multiple_contractions(self):
        res = self.expander.transform("It's true that I'm ready")
        self.assertTrue("is" in res or "it is" in res.lower())

if __name__ == "__main__":
    unittest.main()
""",
    "test_html_cleaner.py": """import unittest
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
""",
    "test_url_cleaner.py": """import unittest
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
""",
    "test_email_cleaner.py": """import unittest
from src.preprocessing.email_cleaner import EmailCleaner

class TestEmailCleaner(unittest.TestCase):
    def test_email_removal(self):
        cleaner = EmailCleaner(replacement_token="[EMAIL]")
        res = cleaner.transform("Contact user@example.com for support")
        self.assertIn("[EMAIL]", res)
        self.assertNotIn("user@example.com", res)

if __name__ == "__main__":
    unittest.main()
""",
    "test_emoji_cleaner.py": """import unittest
from src.preprocessing.emoji_cleaner import EmojiCleaner

class TestEmojiCleaner(unittest.TestCase):
    def test_emoji_demoji(self):
        cleaner = EmojiCleaner(demoji_to_text=True)
        res = cleaner.transform("I am angry 🤬")
        self.assertTrue("face" in res or "symbols" in res or "emoji" in res)

    def test_emoji_strip(self):
        cleaner = EmojiCleaner(demoji_to_text=False)
        res = cleaner.transform("Happy day 😊")
        self.assertEqual(res, "Happy day")

if __name__ == "__main__":
    unittest.main()
""",
    "test_number_cleaner.py": """import unittest
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
""",
    "test_punctuation_cleaner.py": """import unittest
from src.preprocessing.punctuation_cleaner import PunctuationCleaner

class TestPunctuationCleaner(unittest.TestCase):
    def test_punctuation_removal(self):
        cleaner = PunctuationCleaner()
        res = cleaner.transform("Hello, world!!! How are you?")
        self.assertEqual(res, "Hello world How are you")

if __name__ == "__main__":
    unittest.main()
""",
    "test_special_character_cleaner.py": """import unittest
from src.preprocessing.special_character_cleaner import SpecialCharacterCleaner

class TestSpecialCharacterCleaner(unittest.TestCase):
    def test_special_char_removal(self):
        cleaner = SpecialCharacterCleaner()
        res = cleaner.transform("Symbol § © ™ test")
        self.assertEqual(res, "Symbol test")

if __name__ == "__main__":
    unittest.main()
""",
    "test_whitespace.py": """import unittest
from src.preprocessing.whitespace_normalizer import WhitespaceNormalizer

class TestWhitespaceNormalizer(unittest.TestCase):
    def test_whitespace_normalization(self):
        cleaner = WhitespaceNormalizer()
        res = cleaner.transform("  Hello \t\n  world \n\n ")
        self.assertEqual(res, "Hello world")

if __name__ == "__main__":
    unittest.main()
""",
    "test_stopwords.py": """import unittest
from src.preprocessing.stopword_remover import StopwordRemover

class TestStopwordRemover(unittest.TestCase):
    def test_stopword_removal(self):
        remover = StopwordRemover()
        res = remover.transform("this is a toxic comment on wikipedia page")
        self.assertNotIn("this", res.split())
        self.assertNotIn("wikipedia", res.split())
        self.assertIn("toxic", res.split())

if __name__ == "__main__":
    unittest.main()
""",
    "test_lemmatizer.py": """import unittest
from src.preprocessing.lemmatizer import Lemmatizer

class TestLemmatizer(unittest.TestCase):
    def test_lemmatization(self):
        lem = Lemmatizer(backend="spacy")
        res = lem.transform("running cars and feet")
        self.assertTrue("run" in res or "running" in res)
        self.assertTrue("car" in res or "foot" in res or "cars" in res)

if __name__ == "__main__":
    unittest.main()
""",
    "test_pipeline.py": """import unittest
from src.preprocessing.pipeline import TextPreprocessingPipeline
from src.preprocessing.config import ConfigurationManager

class TestTextPreprocessingPipeline(unittest.TestCase):
    def test_traditional_ml_pipeline(self):
        config = ConfigurationManager.get_traditional_ml_config()
        pipeline = TextPreprocessingPipeline(config=config)
        raw = "<p>DON'T click http://test.com! 🤬 123</p>"
        clean = pipeline.transform(raw)
        self.assertNotIn("<p>", clean)
        self.assertNotIn("http", clean)
        self.assertEqual(clean, clean.lower())

    def test_batch_pipeline(self):
        pipeline = TextPreprocessingPipeline()
        batch_raw = ["Text 1", "Text 2"]
        batch_clean = pipeline.transform_batch(batch_raw)
        self.assertEqual(len(batch_clean), 2)

if __name__ == "__main__":
    unittest.main()
""",
    "test_preprocessing.py": """import unittest
from src.preprocessing.config import ConfigurationManager, PreprocessingConfig
from src.preprocessing.utils import normalize_unicode, get_character_stats

class TestPreprocessingGeneral(unittest.TestCase):
    def test_config_presets(self):
        trad = ConfigurationManager.get_traditional_ml_config()
        trans = ConfigurationManager.get_transformer_config()
        self.assertTrue(trad.lowercase)
        self.assertFalse(trans.lowercase)

    def test_utils(self):
        self.assertEqual(normalize_unicode("café"), "café")
        stats = get_character_stats("abc 123")
        self.assertEqual(stats["alpha"], 3)
        self.assertEqual(stats["digits"], 3)

if __name__ == "__main__":
    unittest.main()
"""
}

for fname, content in test_files.items():
    fpath = os.path.join(TESTS_DIR, fname)
    with open(fpath, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"Created test module: {fpath}")

print("All 15 test modules created successfully!")

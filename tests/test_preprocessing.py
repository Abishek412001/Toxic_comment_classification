import unittest
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

import unittest
from src.visualization.config import VisualizationConfig
from src.visualization.theme import ThemeManager

class TestVisualizations(unittest.TestCase):
    def test_theme_manager(self):
        config = VisualizationConfig(theme="recruiter")
        ThemeManager.apply_theme(config)
        self.assertEqual(config.theme, "recruiter")

if __name__ == "__main__":
    unittest.main()

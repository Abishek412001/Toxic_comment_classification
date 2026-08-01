import unittest
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

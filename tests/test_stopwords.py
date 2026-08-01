import unittest
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

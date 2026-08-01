import unittest
from src.preprocessing.lemmatizer import Lemmatizer

class TestLemmatizer(unittest.TestCase):
    def test_lemmatization(self):
        lem = Lemmatizer(backend="spacy")
        res = lem.transform("running cars and feet")
        self.assertTrue("run" in res or "running" in res)
        self.assertTrue("car" in res or "foot" in res or "cars" in res)

if __name__ == "__main__":
    unittest.main()

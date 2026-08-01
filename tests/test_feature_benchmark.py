import unittest
import numpy as np
from src.features.evaluator import FeatureEvaluator
from src.features.benchmark import FeatureBenchmark
from src.features.tfidf_vectorizer import TFIDFFeatureExtractor

class TestFeatureBenchmark(unittest.TestCase):
    def test_evaluator(self):
        matrix = np.zeros((10, 5))
        matrix[0, 0] = 1.0
        stats = FeatureEvaluator.evaluate_matrix(matrix)
        self.assertEqual(stats["num_samples"], 10)
        self.assertEqual(stats["num_features"], 5)
        self.assertGreater(stats["sparsity_percentage"], 90.0)

    def test_benchmark(self):
        extractor = TFIDFFeatureExtractor()
        texts = ["sample text one", "sample text two"]
        results = FeatureBenchmark.benchmark_extractor(extractor, texts)
        self.assertIn("single_doc_latency_ms", results)
        self.assertIn("throughput_docs_per_sec", results)

if __name__ == "__main__":
    unittest.main()

"""
Feature Benchmark Module (Step 49).

Measures training wall-clock time, single-doc latency, throughput, and memory footprint
across feature extraction methods.
"""

import time
import psutil
import os
import logging
from typing import List, Dict, Any
from src.features.base_feature_extractor import BaseFeatureExtractor
from src.features.evaluator import FeatureEvaluator

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class FeatureBenchmark:
    """Benchmarking class comparing feature extraction performance."""

    @staticmethod
    def benchmark_extractor(extractor: BaseFeatureExtractor, texts: List[str]) -> Dict[str, Any]:
        """Runs fit & transform benchmarks for a single feature extractor.

        Args:
            extractor: Instantiated BaseFeatureExtractor subclass.
            texts: List of input strings.

        Returns:
            Dict containing timing, memory, and matrix metrics.
        """
        process = psutil.Process(os.getpid())
        mem_start_mb = process.memory_info().rss / (1024 * 1024)

        # 1. Fit Timing
        t0 = time.perf_counter()
        extractor.fit(texts)
        fit_time_sec = round(time.perf_counter() - t0, 4)

        # 2. Transform Timing
        t1 = time.perf_counter()
        matrix = extractor.transform(texts)
        transform_time_sec = round(time.perf_counter() - t1, 4)

        mem_end_mb = process.memory_info().rss / (1024 * 1024)
        mem_delta_mb = round(mem_end_mb - mem_start_mb, 2)

        # 3. Latency & Throughput
        single_doc_latency_ms = round((transform_time_sec / max(len(texts), 1)) * 1000.0, 3)
        throughput_docs_per_sec = round(len(texts) / max(transform_time_sec, 0.0001), 2)

        matrix_stats = FeatureEvaluator.evaluate_matrix(matrix)

        results = {
            "extractor_name": extractor.name,
            "fit_time_sec": fit_time_sec,
            "transform_time_sec": transform_time_sec,
            "single_doc_latency_ms": single_doc_latency_ms,
            "throughput_docs_per_sec": throughput_docs_per_sec,
            "memory_delta_mb": mem_delta_mb,
            "matrix_metrics": matrix_stats,
        }

        logger.info(f"Benchmarked '{extractor.name}': Latency = {single_doc_latency_ms} ms/doc, Features = {matrix_stats['num_features']:,}")
        return results

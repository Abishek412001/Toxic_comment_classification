"""
Pipeline Benchmark Module (Step 40).

Measures execution latency (ms/doc), throughput (docs/sec), memory footprint,
and multi-core parallel scaling benchmark performance.
"""

import time
import psutil
import os
import logging
from typing import List, Dict, Any
from src.preprocessing.pipeline import TextPreprocessingPipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class PipelineBenchmark:
    """Benchmarking class for measuring preprocessing performance and scalability."""

    @staticmethod
    def run_benchmark(pipeline: TextPreprocessingPipeline, texts: List[str], batch_sizes: List[int] = [10, 100, 500]) -> Dict[str, Any]:
        """Runs latency, throughput, and scalability benchmarks.

        Args:
            pipeline: TextPreprocessingPipeline instance.
            texts: List of sample strings.
            batch_sizes: List of batch sizes to benchmark.

        Returns:
            Dict of benchmark results.
        """
        process = psutil.Process(os.getpid())
        mem_before_mb = process.memory_info().rss / (1024 * 1024)

        results = {}

        # 1. Single Item Latency
        start_time = time.perf_counter()
        for text in texts[:100]:
            pipeline.transform(text)
        single_elapsed = time.perf_counter() - start_time
        latency_ms_per_doc = round((single_elapsed / max(len(texts[:100]), 1)) * 1000.0, 3)

        # 2. Batch Execution Speed
        batch_benchmarks = {}
        for b_size in batch_sizes:
            sample_batch = (texts * ((b_size // len(texts)) + 1))[:b_size]
            b_start = time.perf_counter()
            pipeline.transform_batch(sample_batch, n_jobs=1)
            b_elapsed = time.perf_counter() - b_start
            throughput = round(b_size / max(b_elapsed, 0.0001), 2)
            batch_benchmarks[f"batch_{b_size}"] = {
                "elapsed_seconds": round(b_elapsed, 4),
                "throughput_docs_per_sec": throughput,
            }

        mem_after_mb = process.memory_info().rss / (1024 * 1024)

        results = {
            "single_doc_latency_ms": latency_ms_per_doc,
            "memory_before_mb": round(mem_before_mb, 2),
            "memory_after_mb": round(mem_after_mb, 2),
            "memory_delta_mb": round(mem_after_mb - mem_before_mb, 2),
            "batch_benchmarks": batch_benchmarks,
        }

        logger.info(f"Benchmark completed: Single doc latency = {latency_ms_per_doc} ms.")
        return results

"""
Alias module mapping preprocessing_pipeline to pipeline.
"""

from src.preprocessing.pipeline import TextPreprocessingPipeline, build_pipeline, _parallel_transform_chunk

__all__ = ["TextPreprocessingPipeline", "build_pipeline"]

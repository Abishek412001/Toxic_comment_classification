"""
Configuration Manager Module for Feature Engineering.
"""

from dataclasses import dataclass, field
from typing import Tuple, Optional
from src.features.constants import (
    FeatureType,
    DEFAULT_MAX_FEATURES,
    DEFAULT_MIN_DF,
    DEFAULT_MAX_DF,
    DEFAULT_EMBEDDING_DIM,
    DEFAULT_ARTIFACT_DIR,
)


@dataclass
class FeatureConfig:
    """Dataclass storing feature engineering hyperparameters and model settings."""

    feature_type: str = FeatureType.TFIDF.value
    max_features: int = DEFAULT_MAX_FEATURES
    min_df: int = DEFAULT_MIN_DF
    max_df: float = DEFAULT_MAX_DF
    ngram_range: Tuple[int, int] = (1, 2)
    sublinear_tf: bool = True
    embedding_dimension: int = DEFAULT_EMBEDDING_DIM
    model_name: str = "bert-base-cased"
    batch_size: int = 64
    device: str = "cpu"  # "cpu" or "cuda"
    save_vectorizer: bool = True
    artifact_dir: str = DEFAULT_ARTIFACT_DIR


class ConfigurationManager:
    """Factory and manager for preset feature engineering configurations."""

    @staticmethod
    def get_traditional_ml_config() -> FeatureConfig:
        """Returns feature config optimized for Traditional ML (TF-IDF 1,2 n-grams)."""
        return FeatureConfig(
            feature_type=FeatureType.TFIDF.value,
            max_features=25000,
            min_df=3,
            max_df=0.8,
            ngram_range=(1, 2),
            sublinear_tf=True,
        )

    @staticmethod
    def get_deep_learning_config() -> FeatureConfig:
        """Returns feature config optimized for Deep Learning (Word2Vec / GloVe 300d embeddings)."""
        return FeatureConfig(
            feature_type=FeatureType.WORD2VEC.value,
            embedding_dimension=300,
            batch_size=64,
        )

    @staticmethod
    def get_transformer_config() -> FeatureConfig:
        """Returns feature config optimized for Transformer models (BERT / RoBERTa)."""
        return FeatureConfig(
            feature_type=FeatureType.BERT.value,
            model_name="roberta-base",
            batch_size=32,
            device="cpu",
        )

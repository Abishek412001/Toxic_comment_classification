"""
Constants Module for Feature Engineering.
"""

from enum import Enum


class FeatureType(str, Enum):
    """Enumeration of supported feature extraction architectures."""

    BAG_OF_WORDS = "bag_of_words"
    TFIDF = "tfidf"
    WORD2VEC = "word2vec"
    FASTTEXT = "fasttext"
    GLOVE = "glove"
    BERT = "bert"
    SENTENCE_TRANSFORMER = "sentence_transformer"


DEFAULT_ARTIFACT_DIR = "outputs/models/features"
DEFAULT_MAX_FEATURES = 25000
DEFAULT_MIN_DF = 3
DEFAULT_MAX_DF = 0.8
DEFAULT_EMBEDDING_DIM = 300

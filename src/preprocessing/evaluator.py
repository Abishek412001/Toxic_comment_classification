"""
Pipeline Evaluator Module (Step 40).

Evaluates text statistical changes before vs after preprocessing:
token count reduction %, vocabulary compression %, character reduction %, and legibility metrics.
"""

import logging
from typing import List, Dict, Any
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class PipelineEvaluator:
    """Evaluator class for computing dataset-level text preprocessing statistics."""

    @staticmethod
    def evaluate_texts(raw_texts: List[str], cleaned_texts: List[str]) -> Dict[str, Any]:
        """Computes comparative evaluation statistics.

        Args:
            raw_texts: List of raw comment strings.
            cleaned_texts: List of preprocessed clean strings.

        Returns:
            Dict containing token reduction, vocab compression, and character metrics.
        """
        raw_series = pd.Series(raw_texts).fillna("").astype(str)
        clean_series = pd.Series(cleaned_texts).fillna("").astype(str)

        # Character Statistics
        raw_chars = raw_series.str.len().sum()
        clean_chars = clean_series.str.len().sum()
        char_reduction_pct = round(((raw_chars - clean_chars) / max(raw_chars, 1)) * 100.0, 2)

        # Token Statistics
        raw_tokens = raw_series.str.split().explode().dropna()
        clean_tokens = clean_series.str.split().explode().dropna()

        total_raw_tokens = len(raw_tokens)
        total_clean_tokens = len(clean_tokens)
        token_reduction_pct = round(((total_raw_tokens - total_clean_tokens) / max(total_raw_tokens, 1)) * 100.0, 2)

        # Vocabulary Statistics
        raw_vocab = set(raw_tokens.str.lower())
        clean_vocab = set(clean_tokens.str.lower())

        raw_vocab_size = len(raw_vocab)
        clean_vocab_size = len(clean_vocab)
        vocab_reduction_pct = round(((raw_vocab_size - clean_vocab_size) / max(raw_vocab_size, 1)) * 100.0, 2)

        metrics = {
            "total_raw_characters": int(raw_chars),
            "total_clean_characters": int(clean_chars),
            "character_reduction_pct": char_reduction_pct,
            "total_raw_tokens": int(total_raw_tokens),
            "total_clean_tokens": int(total_clean_tokens),
            "token_reduction_pct": token_reduction_pct,
            "raw_vocabulary_size": int(raw_vocab_size),
            "clean_vocabulary_size": int(clean_vocab_size),
            "vocabulary_reduction_pct": vocab_reduction_pct,
            "avg_raw_token_length": round(raw_series.str.split().str.len().mean(), 2),
            "avg_clean_token_length": round(clean_series.str.split().str.len().mean(), 2),
        }

        logger.info(f"Evaluated pipeline: Token Reduction = {token_reduction_pct}%, Vocab Compression = {vocab_reduction_pct}%")
        return metrics

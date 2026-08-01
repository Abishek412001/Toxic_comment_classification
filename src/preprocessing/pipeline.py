"""
Master Configurable Text Preprocessing Pipeline Module (Step 39).

Assembles all 13 preprocessing stages sequentially with full config toggle support,
single-item transformation, batch processing, and multi-core parallel execution.
"""

import os
import time
import logging
from typing import List, Optional, Union
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed

from src.preprocessing.config import PreprocessingConfig, ConfigurationManager
from src.preprocessing.validator import TextValidator
from src.preprocessing.lowercase import LowercaseTransformer
from src.preprocessing.contractions import ContractionExpander
from src.preprocessing.html_cleaner import HTMLCleaner
from src.preprocessing.url_cleaner import URLCleaner
from src.preprocessing.email_cleaner import EmailCleaner
from src.preprocessing.emoji_cleaner import EmojiCleaner
from src.preprocessing.number_cleaner import NumberCleaner
from src.preprocessing.punctuation_cleaner import PunctuationCleaner
from src.preprocessing.special_character_cleaner import SpecialCharacterCleaner
from src.preprocessing.whitespace_normalizer import WhitespaceNormalizer
from src.preprocessing.stopword_remover import StopwordRemover
from src.preprocessing.lemmatizer import Lemmatizer
from src.preprocessing.exceptions import PreprocessingError, InvalidInputError

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def _parallel_transform_chunk(chunk: List[str], config_dict: dict) -> List[str]:
    """Top-level helper function for multiprocessing worker execution."""
    config = PreprocessingConfig(**config_dict)
    pipeline = TextPreprocessingPipeline(config=config)
    return [pipeline.transform(text) for text in chunk]


class TextPreprocessingPipeline:
    """Master production-grade 13-stage text preprocessing pipeline."""

    def __init__(self, config: Optional[PreprocessingConfig] = None):
        """Initializes pipeline with supplied or default configuration.

        Args:
            config: PreprocessingConfig instance.
        """
        self.config = config or ConfigurationManager.get_traditional_ml_config()
        self._build_pipeline_stages()

    def _build_pipeline_stages(self) -> None:
        """Instantiates registered transformers according to active config flags."""
        self.stages = []

        if self.config.lowercase:
            self.stages.append(("lowercasing", LowercaseTransformer()))

        if self.config.expand_contractions:
            self.stages.append(("contraction_expansion", ContractionExpander()))

        if self.config.remove_html:
            self.stages.append(("html_removal", HTMLCleaner()))

        if self.config.remove_urls:
            self.stages.append(("url_removal", URLCleaner(replacement_token=self.config.url_replacement_token)))

        if self.config.remove_emails:
            self.stages.append(("email_removal", EmailCleaner(replacement_token=self.config.email_replacement_token)))

        if self.config.remove_emojis or self.config.demoji_to_text:
            self.stages.append(("emoji_processing", EmojiCleaner(demoji_to_text=self.config.demoji_to_text)))

        if self.config.remove_numbers or self.config.replace_numbers_with_zero:
            self.stages.append((
                "number_processing",
                NumberCleaner(
                    replacement_token=self.config.number_replacement_token,
                    remove_entirely=self.config.remove_numbers,
                )
            ))

        if self.config.remove_punctuation:
            self.stages.append(("punctuation_removal", PunctuationCleaner()))

        if self.config.remove_special_characters:
            self.stages.append(("special_character_removal", SpecialCharacterCleaner()))

        if self.config.normalize_whitespace:
            self.stages.append(("whitespace_normalization", WhitespaceNormalizer()))

        if self.config.remove_stopwords:
            self.stages.append(("stopword_removal", StopwordRemover(custom_stopwords=self.config.custom_stopwords)))

        if self.config.lemmatization:
            self.stages.append(("lemmatization", Lemmatizer(backend=self.config.lemmatizer_backend)))

        logger.info(f"Built Preprocessing Pipeline with {len(self.stages)} active stages.")

    def transform(self, text: str) -> str:
        """Transforms a single text string through all active pipeline stages.

        Args:
            text: Raw input string.

        Returns:
            Fully preprocessed clean text string.
        """
        validated_text = TextValidator.validate_text(text, allow_empty=True)
        if not validated_text:
            return ""

        current_text = validated_text
        for stage_name, stage_transformer in self.stages:
            current_text = stage_transformer.transform(current_text)

        return current_text

    def transform_batch(self, texts: List[str], n_jobs: int = 1) -> List[str]:
        """Transforms a list of text strings sequentially or in parallel.

        Args:
            texts: List of raw input strings.
            n_jobs: Number of parallel CPU workers (1 for sequential).

        Returns:
            List of clean text strings.
        """
        if not isinstance(texts, (list, tuple, pd.Series)):
            raise InvalidInputError("Input to transform_batch must be a list, tuple, or pandas Series.")

        text_list = list(texts)
        if len(text_list) == 0:
            return []

        if n_jobs == 1 or len(text_list) < 50:
            logger.info(f"Processing batch of {len(text_list):,} comments sequentially...")
            return [self.transform(t) for t in text_list]

        # Multi-core parallel execution
        workers = min(n_jobs if n_jobs > 0 else os.cpu_count() or 4, os.cpu_count() or 4)
        chunk_size = max(1, len(text_list) // workers)
        chunks = [text_list[i : i + chunk_size] for i in range(0, len(text_list), chunk_size)]

        config_dict = {
            k: v for k, v in self.config.__dict__.items() if isinstance(v, (bool, str, int, float))
        }

        results = [None] * len(chunks)
        logger.info(f"Processing batch of {len(text_list):,} comments using {workers} parallel CPU workers...")

        with ProcessPoolExecutor(max_workers=workers) as executor:
            future_to_idx = {
                executor.submit(_parallel_transform_chunk, chunk, config_dict): idx
                for idx, chunk in enumerate(chunks)
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                results[idx] = future.result()

        flat_results = []
        for r in results:
            if r:
                flat_results.extend(r)

        return flat_results


def build_pipeline(preset: str = "traditional_ml") -> TextPreprocessingPipeline:
    """Factory helper to build pipeline instances from preset names."""
    if preset == "traditional_ml":
        cfg = ConfigurationManager.get_traditional_ml_config()
    elif preset == "deep_learning":
        cfg = ConfigurationManager.get_deep_learning_config()
    elif preset == "transformer":
        cfg = ConfigurationManager.get_transformer_config()
    else:
        cfg = PreprocessingConfig()

    return TextPreprocessingPipeline(config=cfg)

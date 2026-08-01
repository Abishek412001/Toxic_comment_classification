"""
Configuration Manager Module for Text Preprocessing.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional
from src.preprocessing.exceptions import ConfigurationError


@dataclass
class PreprocessingConfig:
    """Dataclass holding boolean flags and options for preprocessing steps."""

    lowercase: bool = True
    expand_contractions: bool = True
    remove_html: bool = True
    remove_urls: bool = True
    remove_emails: bool = True
    remove_emojis: bool = False
    demoji_to_text: bool = True
    remove_numbers: bool = False
    replace_numbers_with_zero: bool = True
    remove_punctuation: bool = False
    remove_special_characters: bool = True
    normalize_whitespace: bool = True
    remove_stopwords: bool = True
    lemmatization: bool = True

    url_replacement_token: str = "[URL]"
    email_replacement_token: str = "[EMAIL]"
    number_replacement_token: str = "0"
    lemmatizer_backend: str = "spacy"  # "spacy" or "wordnet"
    custom_stopwords: Optional[set] = field(default_factory=set)


class ConfigurationManager:
    """Factory and manager for preset preprocessing configurations."""

    @staticmethod
    def get_traditional_ml_config() -> PreprocessingConfig:
        """Returns config optimized for Traditional ML (TF-IDF + Logistic Regression/XGBoost)."""
        return PreprocessingConfig(
            lowercase=True,
            expand_contractions=True,
            remove_html=True,
            remove_urls=True,
            remove_emails=True,
            remove_emojis=False,
            demoji_to_text=True,
            remove_numbers=True,
            replace_numbers_with_zero=False,
            remove_punctuation=True,
            remove_special_characters=True,
            normalize_whitespace=True,
            remove_stopwords=True,
            lemmatization=True,
            url_replacement_token="",
            email_replacement_token="",
        )

    @staticmethod
    def get_deep_learning_config() -> PreprocessingConfig:
        """Returns config optimized for Deep Learning (BiLSTM / GloVe Embeddings)."""
        return PreprocessingConfig(
            lowercase=True,
            expand_contractions=True,
            remove_html=True,
            remove_urls=True,
            remove_emails=True,
            remove_emojis=False,
            demoji_to_text=True,
            remove_numbers=False,
            replace_numbers_with_zero=True,
            remove_punctuation=False,
            remove_special_characters=False,
            normalize_whitespace=True,
            remove_stopwords=False,
            lemmatization=False,
            url_replacement_token="[URL]",
            email_replacement_token="[EMAIL]",
        )

    @staticmethod
    def get_transformer_config() -> PreprocessingConfig:
        """Returns config optimized for Transformer Models (BERT / RoBERTa cased models)."""
        return PreprocessingConfig(
            lowercase=False,  # Retain cased text for cased transformers
            expand_contractions=False,  # Subword tokenizers handle contractions
            remove_html=True,
            remove_urls=True,
            remove_emails=True,
            remove_emojis=False,
            demoji_to_text=False,  # Transformers support emoji tokens
            remove_numbers=False,
            replace_numbers_with_zero=False,
            remove_punctuation=False,
            remove_special_characters=False,
            normalize_whitespace=True,
            remove_stopwords=False,
            lemmatization=False,
            url_replacement_token="[URL]",
            email_replacement_token="[EMAIL]",
        )

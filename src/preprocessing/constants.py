"""
Constants & Regex Patterns Module for Text Preprocessing.
"""

import re

# Maximum allowed text length (100,000 characters) to prevent memory DOS
MAX_TEXT_LENGTH = 100000

# Regular Expression Patterns
URL_REGEX = re.compile(
    r"(?:https?://|ftp://|www\.)[^\s/$.?#].[^\s]*",
    re.IGNORECASE,
)

EMAIL_REGEX = re.compile(
    r"\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b",
    re.IGNORECASE,
)

HTML_TAG_REGEX = re.compile(r"<[^>]+>")

NUMBER_REGEX = re.compile(r"\b\d+(?:\.\d+)?\b")

PUNCTUATION_REGEX = re.compile(r"[^\w\s]")

WHITESPACE_REGEX = re.compile(r"\s+")

# Default Wikipedia Domain Stopwords
DEFAULT_DOMAIN_STOPWORDS = {
    "talk", "page", "edit", "wikipedia", "article", "user", "one", "like", "also",
    "see", "make", "know", "think", "people", "use", "time", "way", "even", "first",
}

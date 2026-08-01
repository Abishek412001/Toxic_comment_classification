"""
Custom Exception Hierarchy for Sentiment Analysis Framework (Phase 7).
"""


class SentimentError(Exception):
    """Base exception class for all sentiment framework errors."""
    pass


class ValidationError(SentimentError):
    """Raised when input text sequences fail validation."""
    pass


class SentimentAnalysisError(SentimentError):
    """Raised when sentiment analysis engines fail during execution."""
    pass


class ConfigurationError(SentimentError):
    """Raised when invalid sentiment configuration options are provided."""
    pass

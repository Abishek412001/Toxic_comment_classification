"""
Custom Exceptions Module for Text Preprocessing.

Defines custom exception hierarchy for validation, configuration, input, and cleaning errors.
"""


class PreprocessingError(Exception):
    """Base exception class for all text preprocessing errors."""
    pass


class InvalidInputError(PreprocessingError):
    """Raised when text input fails type, null, or structure validation."""
    pass


class EmptyTextError(PreprocessingError):
    """Raised when text input is empty or whitespace-only when non-empty text is required."""
    pass


class ConfigurationError(PreprocessingError):
    """Raised when invalid or conflicting preprocessing configurations are supplied."""
    pass


class CleaningError(PreprocessingError):
    """Raised when an internal error occurs during a specific cleaning transformer step."""
    pass

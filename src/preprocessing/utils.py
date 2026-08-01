"""
Preprocessing Utility Functions Module.
"""

import unicodedata


def normalize_unicode(text: str, form: str = "NFKC") -> str:
    """Normalizes Unicode text to standard form (NFC / NFKC).

    Args:
        text: Input string.
        form: Normalization form.

    Returns:
        Normalized string.
    """
    if not isinstance(text, str):
        return ""
    return unicodedata.normalize(form, text)


def get_character_stats(text: str) -> dict:
    """Computes basic character composition statistics.

    Args:
        text: Input string.

    Returns:
        Dict of character metrics.
    """
    if not text:
        return {"length": 0, "alpha": 0, "digits": 0, "spaces": 0}

    return {
        "length": len(text),
        "alpha": sum(c.isalpha() for c in text),
        "digits": sum(c.isdigit() for c in text),
        "spaces": sum(c.isspace() for c in text),
    }

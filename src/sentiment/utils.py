"""
Sentiment Utilities Module.

Provides compound score normalization and label mapping helper functions.
"""

from typing import Dict, Any


def compound_to_label(compound_score: float, pos_thresh: float = 0.05, neg_thresh: float = -0.05) -> str:
    """Maps a numerical compound score to positive, neutral, or negative.

    Args:
        compound_score: Numerical float in [-1.0, +1.0].
        pos_thresh: Positive cutoff bound.
        neg_thresh: Negative cutoff bound.

    Returns:
        Sentiment label string ('positive', 'neutral', 'negative').
    """
    if compound_score >= pos_thresh:
        return "positive"
    elif compound_score <= neg_thresh:
        return "negative"
    else:
        return "neutral"

"""
XAI Utilities Module.

Provides feature importance sorting, positive/negative contributor separation, and text highlighting helpers.
"""

from typing import Dict, List, Tuple, Any


def split_positive_negative_contributors(feature_importance: Dict[str, float]) -> Tuple[List[Tuple[str, float]], List[Tuple[str, float]]]:
    """Splits word feature importance dictionary into positive (increasing toxicity) and negative (decreasing toxicity) lists.

    Args:
        feature_importance: Dictionary mapping word tokens to weight floats.

    Returns:
        Tuple of (positive_contributors, negative_contributors) sorted by magnitude.
    """
    positives = [(k, v) for k, v in feature_importance.items() if v > 0]
    negatives = [(k, v) for k, v in feature_importance.items() if v < 0]

    positives_sorted = sorted(positives, key=lambda item: abs(item[1]), reverse=True)
    negatives_sorted = sorted(negatives, key=lambda item: abs(item[1]), reverse=True)

    return positives_sorted, negatives_sorted

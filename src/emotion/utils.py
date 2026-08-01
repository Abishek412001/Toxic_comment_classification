"""
Emotion Utilities Module.

Provides top-k emotion ranking and probability distribution normalization functions.
"""

from typing import Dict, List, Tuple


def get_top_k_emotions(probabilities: Dict[str, float], top_k: int = 3) -> List[Tuple[str, float]]:
    """Extracts top-k emotion labels and probability values sorted descending.

    Args:
        probabilities: Dictionary mapping emotion labels to probability floats.
        top_k: Number of top emotions to extract.

    Returns:
        List of tuples (emotion_label, probability).
    """
    sorted_emotions = sorted(probabilities.items(), key=lambda item: item[1], reverse=True)
    return sorted_emotions[:top_k]

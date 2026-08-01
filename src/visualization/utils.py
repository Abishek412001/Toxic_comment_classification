"""
Visualization Utilities Module.

Provides color conversions, data scaling, and layout formatting helpers.
"""

from typing import List, Tuple, Dict


def get_recruiter_color_palette() -> List[str]:
    """Returns curated 7-color palette for recruiter-grade dashboards.

    Returns:
        List of hex color strings.
    """
    return ["#2c3e50", "#1abc9c", "#e74c3c", "#3498db", "#f1c40f", "#8e44ad", "#e67e22"]

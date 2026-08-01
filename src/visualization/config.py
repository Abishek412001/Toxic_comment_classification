"""
Configuration Manager Module for Analytics & Visualization.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from src.visualization.constants import THEMES, DEFAULT_DPI, DEFAULT_FIGSIZE, DEFAULT_VISUALIZATION_DIR


@dataclass
class VisualizationConfig:
    """Dataclass storing theme preferences, figure dimensions, DPI settings, and export paths."""

    theme: str = "recruiter"  # "dark", "light", "recruiter"
    dpi: int = DEFAULT_DPI
    figsize: tuple = DEFAULT_FIGSIZE
    output_dir: str = DEFAULT_VISUALIZATION_DIR
    save_png: bool = True
    save_html: bool = True
    save_pdf: bool = True

"""
Custom Exception Hierarchy for Analytics & Visualization Framework (Phase 10).
"""


class VisualizationError(Exception):
    """Base exception class for all visualization framework errors."""
    pass


class ChartError(VisualizationError):
    """Raised when chart building or rendering fails."""
    pass


class ThemeError(VisualizationError):
    """Raised when invalid theme or color palette configurations are provided."""
    pass


class ExportError(VisualizationError):
    """Raised when HTML, PNG, or PDF figure export fails."""
    pass

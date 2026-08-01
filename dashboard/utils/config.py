"""
Dashboard Config Utility.
"""

from dataclasses import dataclass

@dataclass
class DashboardConfig:
    title: str = "Toxic Comment Classification & Intelligence System"
    icon: str = "🛡️"
    layout: str = "wide"
    default_model: str = "distilbert"
    css_path: str = "dashboard/assets/css/style.css"

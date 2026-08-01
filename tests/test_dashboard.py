import unittest
from src.visualization.kpi_dashboard import KPIManager

class TestDashboard(unittest.TestCase):
    def test_executive_kpis(self):
        kpis = KPIManager.get_executive_kpis()
        self.assertIn("overall_toxicity_rate", kpis)
        self.assertIn("champion_model", kpis)

if __name__ == "__main__":
    unittest.main()

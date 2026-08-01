import unittest
import pandas as pd
from src.visualization.plotly_charts import PlotlyVisualizer

class TestPlotlyVisualizer(unittest.TestCase):
    def test_create_bar_chart(self):
        df = pd.DataFrame({"x": ["A", "B"], "y": [10, 20]})
        fig = PlotlyVisualizer.create_bar_chart(df, "x", "y", "Test Bar")
        self.assertIsNotNone(fig)

if __name__ == "__main__":
    unittest.main()

import unittest
from src.mlops.environment import Environment
from src.mlops.settings import Settings
from src.mlops.health import HealthChecker
from src.mlops.model_loader import ModelLoader

class TestMLOps(unittest.TestCase):
    def test_environment(self):
        env = Environment.get_env()
        self.assertIn(env, ["development", "testing", "staging", "production"])

    def test_settings(self):
        settings = Settings()
        val = settings.get("debug", True)
        self.assertIsNotNone(val)

    def test_health_checker(self):
        live = HealthChecker.check_liveness()
        self.assertEqual(live["status"], "UP")

        ready = HealthChecker.check_readiness()
        self.assertIn(ready["status"], ["UP", "DOWN"])

    def test_model_loader(self):
        m = ModelLoader.load_model("mock_m", lambda: "model_obj")
        self.assertEqual(m, "model_obj")

if __name__ == "__main__":
    unittest.main()

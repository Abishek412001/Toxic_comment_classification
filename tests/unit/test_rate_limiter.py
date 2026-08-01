"""
Unit Tests for Sliding Window Rate Limiter.
"""

import unittest
from opentrust_core.auth.rate_limiter import SlidingWindowRateLimiter
from opentrust_core.auth.models import PlanTierEnum
from opentrust_core.exceptions import RateLimitExceededError


class TestRateLimiter(unittest.TestCase):
    def setUp(self):
        self.limiter = SlidingWindowRateLimiter()

    def test_free_tier_rate_limit(self):
        client = "client_free_test"
        for _ in range(60):
            remaining = self.limiter.check_rate_limit(client, plan_tier=PlanTierEnum.FREE)
            self.assertGreaterEqual(remaining, 0)

        with self.assertRaises(RateLimitExceededError):
            self.limiter.check_rate_limit(client, plan_tier=PlanTierEnum.FREE)

    def test_enterprise_tier_limit(self):
        client = "client_ent_test"
        remaining = self.limiter.check_rate_limit(client, plan_tier=PlanTierEnum.ENTERPRISE)
        self.assertEqual(remaining, 5999)


if __name__ == "__main__":
    unittest.main()

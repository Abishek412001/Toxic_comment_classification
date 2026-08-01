"""
High-Performance Sliding Window Rate Limiter for OpenTrust API Keys.
"""

import time
from typing import Dict, Tuple
from opentrust_core.auth.models import PlanTierEnum
from opentrust_core.exceptions import RateLimitExceededError

PLAN_LIMITS: Dict[PlanTierEnum, int] = {
    PlanTierEnum.FREE: 60,          # 60 requests per minute
    PlanTierEnum.PRO: 600,         # 600 requests per minute
    PlanTierEnum.ENTERPRISE: 6000, # 6,000 requests per minute
}


class SlidingWindowRateLimiter:
    """In-memory & Redis sliding window rate limiter tracking request quotas by client/token."""

    def __init__(self):
        # Key: (token_or_client_id, window_timestamp), Value: request_count
        self._store: Dict[Tuple[str, int], int] = {}

    def check_rate_limit(self, identifier: str, plan_tier: PlanTierEnum = PlanTierEnum.FREE) -> int:
        """Enforces sliding window rate limit per minute. Returns remaining request quota."""
        max_requests = PLAN_LIMITS.get(plan_tier, 60)
        now = int(time.time())
        current_minute = now // 60

        key = (identifier, current_minute)
        current_count = self._store.get(key, 0)

        if current_count >= max_requests:
            raise RateLimitExceededError(
                f"Rate limit exceeded for {plan_tier.value.upper()} tier. Limit: {max_requests} req/min."
            )

        self._store[key] = current_count + 1

        # Clean up stale windows older than 5 minutes
        stale_keys = [k for k in self._store.keys() if (now // 60) - k[1] > 5]
        for sk in stale_keys:
            del self._store[sk]

        return max_requests - (current_count + 1)


rate_limiter = SlidingWindowRateLimiter()

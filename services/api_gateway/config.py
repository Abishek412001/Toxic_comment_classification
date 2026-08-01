"""
API Gateway Specific Configuration.
"""

from opentrust_core.config import settings

class GatewayConfig:
    SERVICE_NAME: str = "API Gateway"
    VERSION: str = settings.VERSION
    API_PREFIX: str = settings.API_PREFIX

gateway_config = GatewayConfig()

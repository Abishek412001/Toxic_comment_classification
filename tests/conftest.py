"""
Pytest Fixtures for OpenTrust AI Test Suite.
"""

import os
import sys
import pytest
from fastapi.testclient import TestClient

# Ensure root directory is in sys.path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from services.api_gateway.app import app


@pytest.fixture
def client():
    """Returns FastAPI TestClient for Gateway Integration Testing."""
    return TestClient(app)

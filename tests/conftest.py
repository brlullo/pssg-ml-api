import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.models.registry import registry


@pytest.fixture(scope="session", autouse=True)
def load_models():
    """Ensure models are loaded before any tests run."""
    registry.load_all()


@pytest.fixture()
def client():
    with TestClient(app) as c:
        yield c

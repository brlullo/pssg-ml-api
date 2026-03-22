import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture(scope="session")
def client():
    # The lifespan context loads all models and registers per-model routes.
    with TestClient(app) as c:
        yield c

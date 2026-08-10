import pytest
from fastapi.testclient import TestClient

from app.config import settings
from app.main import app
from app.models.registry import registry


@pytest.fixture(scope="session", autouse=True)
def load_models():
    """Ensure models are loaded before any tests run."""
    registry.load_all()


@pytest.fixture(autouse=True)
def isolate_log_db(tmp_path, monkeypatch):
    """Point logging at a throwaway SQLite file so tests never touch real data.

    Also forces the SQLite code path (DATABASE_URL="") so CI needs no Postgres.
    """
    monkeypatch.setattr(settings, "LOG_DB_PATH", str(tmp_path / "test_logs.db"))
    monkeypatch.setattr(settings, "DATABASE_URL", "")


@pytest.fixture()
def client():
    with TestClient(app) as c:
        yield c

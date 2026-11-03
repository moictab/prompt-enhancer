import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture(autouse=True)
def isolated_data_dir(tmp_path, monkeypatch):
    # Every OpenRouter call writes to the request log under DATA_DIR, including in
    # tests that never build the app: keep all of it out of the repo's data/.
    monkeypatch.setenv("DATA_DIR", str(tmp_path))


@pytest.fixture
def api_client(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("ADMIN_USERNAME", "admin")
    monkeypatch.setenv("ADMIN_PASSWORD", "test-password")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    return TestClient(app)


@pytest.fixture
def auth_headers():
    return ("admin", "test-password")

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture
def api_client(tmp_path, monkeypatch):
    # Page routes rely on startup-time seeding (ensure_seed_data), which only
    # runs when the ASGI lifespan is triggered -- i.e. TestClient used as a
    # context manager. The shared conftest api_client fixture intentionally
    # does not do this (other tests rely on a pristine, unseeded data dir),
    # so this file overrides it locally for these tests only.
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("ADMIN_USERNAME", "admin")
    monkeypatch.setenv("ADMIN_PASSWORD", "test-password")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    with TestClient(app) as client:
        yield client


def test_index_requires_auth(api_client):
    response = api_client.get("/")

    assert response.status_code == 401


def test_index_renders_seeded_families(api_client, auth_headers):
    response = api_client.get("/", auth=auth_headers)

    assert response.status_code == 200
    assert "SDXL" in response.text
    assert "Z-Image-Turbo" in response.text


def test_historial_page_renders(api_client, auth_headers):
    response = api_client.get("/historial", auth=auth_headers)

    assert response.status_code == 200


def test_historial_page_has_history_table(api_client, auth_headers):
    response = api_client.get("/historial", auth=auth_headers)

    assert response.status_code == 200
    assert 'id="history-body"' in response.text


def test_admin_page_renders(api_client, auth_headers):
    response = api_client.get("/admin", auth=auth_headers)

    assert response.status_code == 200


def test_index_renders_all_three_tabs(api_client, auth_headers):
    response = api_client.get("/", auth=auth_headers)

    assert response.status_code == 200
    assert 'id="form-generar"' in response.text
    assert 'id="form-iterar"' in response.text
    assert 'id="form-imagen"' in response.text


def test_admin_page_has_management_sections(api_client, auth_headers):
    response = api_client.get("/admin", auth=auth_headers)

    assert response.status_code == 200
    assert 'id="family-form"' in response.text
    assert 'id="character-form"' in response.text
    assert 'id="system-prompt-text"' in response.text

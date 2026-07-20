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


def test_startup_raises_when_admin_password_unset(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("ADMIN_USERNAME", "admin")
    monkeypatch.setenv("ADMIN_PASSWORD", "")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")

    with pytest.raises(RuntimeError, match="ADMIN_PASSWORD"):
        with TestClient(app) as client:
            client.get("/")


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


def test_index_renders_generar_and_imagen_tabs(api_client, auth_headers):
    response = api_client.get("/", auth=auth_headers)

    assert response.status_code == 200
    assert 'id="form-generar"' in response.text
    assert 'id="form-imagen"' in response.text
    assert 'id="generar-previous-prompt"' in response.text
    assert 'id="form-iterar"' not in response.text


def test_index_llm_model_inputs_use_openrouter_models_datalist(api_client, auth_headers):
    response = api_client.get("/", auth=auth_headers)

    assert response.status_code == 200
    assert 'id="openrouter-models"' in response.text
    assert 'list="openrouter-models"' in response.text


def test_admin_page_has_management_sections(api_client, auth_headers):
    response = api_client.get("/admin", auth=auth_headers)

    assert response.status_code == 200
    assert 'id="family-form"' in response.text
    assert 'id="character-form"' in response.text
    assert 'id="system-prompt-generate-text"' in response.text
    assert 'id="system-prompt-iterate-text"' in response.text
    assert 'id="system-prompt-image-text"' in response.text

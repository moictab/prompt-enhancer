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


def test_the_home_page_is_conversar(api_client, auth_headers):
    response = api_client.get("/", auth=auth_headers, follow_redirects=False)

    assert response.status_code == 307 and response.headers["location"] == "/conversar"
    page = api_client.get("/", auth=auth_headers)
    assert "SDXL" in page.text and "Z-Image-Turbo" in page.text
    assert 'list="openrouter-models"' in page.text and 'id="openrouter-models"' in page.text
    assert ">Crear</a>" not in page.text


def test_historial_page_renders(api_client, auth_headers):
    response = api_client.get("/historial", auth=auth_headers)

    assert response.status_code == 200


def test_historial_page_has_history_list(api_client, auth_headers):
    response = api_client.get("/historial", auth=auth_headers)

    assert response.status_code == 200
    assert 'id="history-list"' in response.text


def test_admin_page_renders(api_client, auth_headers):
    response = api_client.get("/admin", auth=auth_headers)

    assert response.status_code == 200


def test_conversar_page_renders_with_families_and_default_model(api_client, auth_headers, monkeypatch):
    monkeypatch.setenv("DEFAULT_LLM_MODEL", "vendor/chat-model")

    response = api_client.get("/conversar", auth=auth_headers)

    assert response.status_code == 200
    assert 'id="chat-log"' in response.text
    assert 'value="vendor/chat-model"' in response.text
    assert ">SDXL</option>" in response.text
    assert '<a href="/conversar" aria-current="page">' in response.text


def test_peticiones_page_renders_under_ajustes(api_client, auth_headers):
    response = api_client.get("/peticiones", auth=auth_headers)

    assert response.status_code == 200
    assert 'id="request-list"' in response.text
    assert '<a href="/admin" aria-current="page">' in response.text


def test_admin_page_has_management_sections(api_client, auth_headers):
    response = api_client.get("/admin", auth=auth_headers)

    assert response.status_code == 200
    assert 'id="family-form"' in response.text
    assert 'id="system-prompt-text"' in response.text
    assert 'id="memory-list"' in response.text
    assert 'data-mode="conversation"' in response.text
    for mode in ["generate", "image", "story", "variations", "iterate"]:
        assert f'data-mode="{mode}"' not in response.text


def test_nav_marks_the_current_page(api_client, auth_headers):
    response = api_client.get("/historial", auth=auth_headers)

    assert '<a href="/historial" aria-current="page">' in response.text


def test_static_assets_are_served_as_revalidated_javascript(api_client, auth_headers):
    response = api_client.get("/static/common.js", auth=auth_headers)

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/javascript")
    assert response.headers["cache-control"] == "no-cache"

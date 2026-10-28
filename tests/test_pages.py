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


def test_historial_page_has_history_list(api_client, auth_headers):
    response = api_client.get("/historial", auth=auth_headers)

    assert response.status_code == 200
    assert 'id="history-list"' in response.text


def test_admin_page_renders(api_client, auth_headers):
    response = api_client.get("/admin", auth=auth_headers)

    assert response.status_code == 200


def test_index_renders_every_mode_in_one_composer(api_client, auth_headers):
    response = api_client.get("/", auth=auth_headers)

    assert response.status_code == 200
    assert 'id="composer-form"' in response.text
    for mode in ["generate", "image", "story", "variations"]:
        assert f'data-mode="{mode}"' in response.text
    assert 'id="previous-prompt"' in response.text
    assert 'id="takes"' in response.text
    assert 'id="characters-dialog"' in response.text


def test_index_renders_every_variation_axis(api_client, auth_headers):
    from app.prompts import VARIATION_AXES

    response = api_client.get("/", auth=auth_headers)

    for key, label in VARIATION_AXES.items():
        assert f'data-axis="{key}"' in response.text
        assert label in response.text


def test_index_renders_reference_prompts_field_in_variations(api_client, auth_headers):
    response = api_client.get("/", auth=auth_headers)

    assert response.status_code == 200
    assert 'id="references-list"' in response.text
    assert "Prompts de referencia" in response.text


def test_index_prefills_default_models_from_settings(api_client, auth_headers, monkeypatch):
    monkeypatch.setenv("DEFAULT_LLM_MODEL", "vendor/text-model")
    monkeypatch.setenv("DEFAULT_VISION_MODEL", "vendor/vision-model")

    response = api_client.get("/", auth=auth_headers)

    assert 'value="vendor/text-model"' in response.text
    assert 'value="vendor/vision-model"' in response.text


def test_index_llm_model_inputs_use_openrouter_models_datalist(api_client, auth_headers):
    response = api_client.get("/", auth=auth_headers)

    assert response.status_code == 200
    assert 'id="openrouter-models"' in response.text
    assert 'list="openrouter-models"' in response.text


def test_index_vision_model_input_uses_filtered_vision_datalist(api_client, auth_headers):
    response = api_client.get("/", auth=auth_headers)

    assert response.status_code == 200
    assert 'id="openrouter-vision-models"' in response.text
    assert 'list="openrouter-vision-models"' in response.text


def test_admin_page_has_management_sections(api_client, auth_headers):
    response = api_client.get("/admin", auth=auth_headers)

    assert response.status_code == 200
    assert 'id="family-form"' in response.text
    assert 'id="character-form"' in response.text
    assert 'id="system-prompt-text"' in response.text
    for mode in ["generate", "iterate", "image", "story", "variations", "extract_character"]:
        assert f'data-mode="{mode}"' in response.text


def test_nav_marks_the_current_page(api_client, auth_headers):
    response = api_client.get("/historial", auth=auth_headers)

    assert '<a href="/historial" aria-current="page">' in response.text


def test_static_assets_are_served_as_revalidated_javascript(api_client, auth_headers):
    response = api_client.get("/static/common.js", auth=auth_headers)

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/javascript")
    assert response.headers["cache-control"] == "no-cache"

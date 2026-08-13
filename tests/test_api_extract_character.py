import base64
from unittest.mock import patch

from app.openrouter_client import OpenRouterResult

# 1x1 transparent PNG
PNG_BYTES = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk"
    "+A8AAQUBAScY42YAAAAASUVORK5CYII="
)


def test_extract_character_requires_auth(api_client):
    response = api_client.post(
        "/api/extract-character",
        json={"prompt_text": "a stoic ronin", "llm_model": "m"},
    )

    assert response.status_code == 401


def test_extract_character_rejects_blank_prompt_text(api_client, auth_headers):
    response = api_client.post(
        "/api/extract-character",
        json={"prompt_text": "   ", "llm_model": "m"},
        auth=auth_headers,
    )

    assert response.status_code == 400


@patch("app.routes.api.call_openrouter")
def test_extract_character_returns_parsed_name_and_text(mock_call, api_client, auth_headers):
    mock_call.return_value = OpenRouterResult(
        content="NAME: Kaito\nTEXT: a stoic ronin with a scarred left eye", cost=0.000456
    )

    response = api_client.post(
        "/api/extract-character",
        json={"prompt_text": "a cyberpunk samurai named Kaito", "llm_model": "anthropic/claude-sonnet-4"},
        auth=auth_headers,
    )

    assert response.status_code == 200
    assert response.json() == {
        "name": "Kaito",
        "text": "a stoic ronin with a scarred left eye",
        "cost": 0.000456,
    }


@patch("app.routes.api.call_openrouter")
def test_extract_character_surfaces_openrouter_errors_as_502(mock_call, api_client, auth_headers):
    mock_call.side_effect = RuntimeError("Unexpected response format from OpenRouter: ")

    response = api_client.post(
        "/api/extract-character",
        json={"prompt_text": "a cyberpunk samurai", "llm_model": "m"},
        auth=auth_headers,
    )

    assert response.status_code == 502


def test_extract_character_from_image_requires_auth(api_client):
    response = api_client.post(
        "/api/extract-character-from-image",
        files={"image": ("test.png", PNG_BYTES, "image/png")},
        data={"vision_model": "m"},
    )

    assert response.status_code == 401


@patch("app.routes.api.call_openrouter")
def test_extract_character_from_image_returns_parsed_name_and_text(mock_call, api_client, auth_headers):
    mock_call.return_value = OpenRouterResult(
        content="NAME: Mika\nTEXT: a neon-haired hacker in a trench coat", cost=0.000789
    )

    response = api_client.post(
        "/api/extract-character-from-image",
        files={"image": ("test.png", PNG_BYTES, "image/png")},
        data={"vision_model": "anthropic/claude-sonnet-4"},
        auth=auth_headers,
    )

    assert response.status_code == 200
    assert response.json() == {
        "name": "Mika",
        "text": "a neon-haired hacker in a trench coat",
        "cost": 0.000789,
    }


@patch("app.routes.api.call_openrouter")
def test_extract_character_from_image_passes_base64_data_uri(mock_call, api_client, auth_headers):
    mock_call.return_value = OpenRouterResult(content="NAME: ok\nTEXT: ok", cost=None)

    api_client.post(
        "/api/extract-character-from-image",
        files={"image": ("test.png", PNG_BYTES, "image/png")},
        data={"vision_model": "m"},
        auth=auth_headers,
    )

    _, kwargs = mock_call.call_args
    assert kwargs["image_data_uri"].startswith("data:image/png;base64,")


def test_extract_character_from_image_rejects_unsupported_content_type(api_client, auth_headers):
    response = api_client.post(
        "/api/extract-character-from-image",
        files={"image": ("test.gif", b"not-really-a-gif", "image/gif")},
        data={"vision_model": "m"},
        auth=auth_headers,
    )

    assert response.status_code == 400


def test_extract_character_from_image_rejects_oversized_image(api_client, auth_headers):
    oversized = b"0" * (10 * 1024 * 1024 + 1)

    response = api_client.post(
        "/api/extract-character-from-image",
        files={"image": ("big.png", oversized, "image/png")},
        data={"vision_model": "m"},
        auth=auth_headers,
    )

    assert response.status_code == 400

import base64
from unittest.mock import patch

from app import families
from app.openrouter_client import OpenRouterResult

# 1x1 transparent PNG
PNG_BYTES = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk"
    "+A8AAQUBAScY42YAAAAASUVORK5CYII="
)


def _create_family(tmp_path, has_negative_prompt=True):
    return families.create_family(
        "SDXL", "family instructions", has_negative_prompt,
        path=str(tmp_path / "families.json"),
    )


def test_from_image_requires_auth(api_client):
    response = api_client.post(
        "/api/from-image",
        files={"image": ("test.png", PNG_BYTES, "image/png")},
        data={"family_id": "x", "vision_model": "m"},
    )

    assert response.status_code == 401


@patch("app.routes.api.call_openrouter")
def test_from_image_returns_parsed_prompt(mock_call, api_client, auth_headers, tmp_path):
    family = _create_family(tmp_path)
    mock_call.return_value = OpenRouterResult(content="POSITIVE: a mountain landscape\nNEGATIVE: blurry", cost=0.000123)

    response = api_client.post(
        "/api/from-image",
        files={"image": ("test.png", PNG_BYTES, "image/png")},
        data={"family_id": family["id"], "vision_model": "anthropic/claude-sonnet-4"},
        auth=auth_headers,
    )

    assert response.status_code == 200
    assert response.json() == {
        "positive_prompt": "a mountain landscape",
        "negative_prompt": "blurry",
        "cost": 0.000123,
    }


@patch("app.routes.api.call_openrouter")
def test_from_image_passes_base64_data_uri(mock_call, api_client, auth_headers, tmp_path):
    family = _create_family(tmp_path)
    mock_call.return_value = OpenRouterResult(content="POSITIVE: ok\nNEGATIVE: ", cost=0.000123)

    api_client.post(
        "/api/from-image",
        files={"image": ("test.png", PNG_BYTES, "image/png")},
        data={"family_id": family["id"], "vision_model": "m"},
        auth=auth_headers,
    )

    _, kwargs = mock_call.call_args
    assert kwargs["image_data_uri"].startswith("data:image/png;base64,")


def test_from_image_rejects_unsupported_content_type(api_client, auth_headers, tmp_path):
    family = _create_family(tmp_path)

    response = api_client.post(
        "/api/from-image",
        files={"image": ("test.gif", b"not-really-a-gif", "image/gif")},
        data={"family_id": family["id"], "vision_model": "m"},
        auth=auth_headers,
    )

    assert response.status_code == 400


def test_from_image_rejects_oversized_image(api_client, auth_headers, tmp_path):
    family = _create_family(tmp_path)
    oversized = b"0" * (10 * 1024 * 1024 + 1)

    response = api_client.post(
        "/api/from-image",
        files={"image": ("big.png", oversized, "image/png")},
        data={"family_id": family["id"], "vision_model": "m"},
        auth=auth_headers,
    )

    assert response.status_code == 400


@patch("app.routes.api.call_openrouter")
def test_from_image_appends_history_with_vision_model_and_no_llm_model(mock_call, api_client, auth_headers, tmp_path):
    family = _create_family(tmp_path)
    mock_call.return_value = OpenRouterResult(content="POSITIVE: ok\nNEGATIVE: ", cost=0.000123)

    api_client.post(
        "/api/from-image",
        files={"image": ("test.png", PNG_BYTES, "image/png")},
        data={"family_id": family["id"], "vision_model": "anthropic/claude-sonnet-4"},
        auth=auth_headers,
    )

    from app import history
    entries = history.list_entries(path=str(tmp_path / "history.jsonl"))
    assert entries[0]["mode"] == "image"
    assert entries[0]["vision_model"] == "anthropic/claude-sonnet-4"
    assert entries[0]["llm_model"] is None

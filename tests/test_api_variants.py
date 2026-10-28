import base64
from unittest.mock import patch

import pytest

from app import characters, families, history
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


@patch("app.routes.api.call_openrouter")
def test_generate_variants_makes_one_call_per_variant_and_sums_cost(
    mock_call, api_client, auth_headers, tmp_path
):
    family = _create_family(tmp_path)
    mock_call.side_effect = [
        OpenRouterResult(content=f"POSITIVE: cat take {i}\nNEGATIVE: blurry", cost=0.001)
        for i in range(3)
    ]

    response = api_client.post(
        "/api/generate",
        json={"user_input": "a cat", "family_id": family["id"], "llm_model": "m", "variants": 3},
        auth=auth_headers,
    )

    assert response.status_code == 200
    body = response.json()
    assert mock_call.call_count == 3
    assert sorted(p["positive_prompt"] for p in body["prompts"]) == [
        "cat take 0", "cat take 1", "cat take 2",
    ]
    assert body["cost"] == pytest.approx(0.003)
    assert body["count_mismatch"] is False
    sent_messages = sorted(c.kwargs["user_message"] for c in mock_call.call_args_list)
    assert all("## Alternative" in m for m in sent_messages)
    assert "this is number 1" in sent_messages[0]
    assert "this is number 3" in sent_messages[2]


@patch("app.routes.api.call_openrouter")
def test_generate_single_variant_omits_alternative_section(mock_call, api_client, auth_headers, tmp_path):
    family = _create_family(tmp_path)
    mock_call.return_value = OpenRouterResult(content="POSITIVE: a cat\nNEGATIVE: ", cost=0.001)

    api_client.post(
        "/api/generate",
        json={"user_input": "a cat", "family_id": family["id"], "llm_model": "m"},
        auth=auth_headers,
    )

    assert mock_call.call_count == 1
    assert "Alternative" not in mock_call.call_args.kwargs["user_message"]


@patch("app.routes.api.call_openrouter")
def test_generate_variants_keeps_successes_when_some_calls_fail(
    mock_call, api_client, auth_headers, tmp_path
):
    family = _create_family(tmp_path)
    mock_call.side_effect = [
        OpenRouterResult(content="POSITIVE: survivor\nNEGATIVE: ", cost=0.001),
        RuntimeError("OpenRouter rate limit exceeded. Wait a moment and try again."),
    ]

    response = api_client.post(
        "/api/generate",
        json={"user_input": "a cat", "family_id": family["id"], "llm_model": "m", "variants": 2},
        auth=auth_headers,
    )

    assert response.status_code == 200
    body = response.json()
    assert [p["positive_prompt"] for p in body["prompts"]] == ["survivor"]
    assert body["count_mismatch"] is True


@patch("app.routes.api.call_openrouter")
def test_generate_variants_returns_502_when_every_call_fails(
    mock_call, api_client, auth_headers, tmp_path
):
    family = _create_family(tmp_path)
    mock_call.side_effect = RuntimeError("OpenRouter rate limit exceeded. Wait a moment and try again.")

    response = api_client.post(
        "/api/generate",
        json={"user_input": "a cat", "family_id": family["id"], "llm_model": "m", "variants": 2},
        auth=auth_headers,
    )

    assert response.status_code == 502
    assert "rate limit" in response.json()["detail"]
    assert history.list_entries(path=str(tmp_path / "history.jsonl")) == []


def test_generate_rejects_too_many_variants(api_client, auth_headers, tmp_path):
    family = _create_family(tmp_path)

    response = api_client.post(
        "/api/generate",
        json={"user_input": "a cat", "family_id": family["id"], "llm_model": "m", "variants": 5},
        auth=auth_headers,
    )

    assert response.status_code == 422


@patch("app.routes.api.call_openrouter")
def test_generate_returns_the_stored_history_entry(mock_call, api_client, auth_headers, tmp_path):
    family = _create_family(tmp_path)
    character = characters.create_character(
        "Kaito", "a stoic ronin", path=str(tmp_path / "characters.json")
    )
    mock_call.return_value = OpenRouterResult(content="POSITIVE: a cat\nNEGATIVE: ", cost=0.001)

    body = api_client.post(
        "/api/generate",
        json={
            "user_input": "a cat", "family_id": family["id"], "llm_model": "m",
            "character_ids": [character["id"]],
        },
        auth=auth_headers,
    ).json()

    [stored] = history.list_entries(path=str(tmp_path / "history.jsonl"))
    assert body["id"] == stored["id"]
    assert body["mode"] == "generate"
    assert body["family_name"] == "SDXL"
    assert body["favorite"] is False
    assert body["params"] == {"variants": 1, "character_names": ["Kaito"]}
    assert stored["cost"] == 0.001


@patch("app.routes.api.call_openrouter")
def test_from_image_variants_send_the_image_with_every_call(mock_call, api_client, auth_headers, tmp_path):
    family = _create_family(tmp_path)
    mock_call.return_value = OpenRouterResult(content="POSITIVE: ok\nNEGATIVE: ", cost=0.001)

    response = api_client.post(
        "/api/from-image",
        files={"image": ("test.png", PNG_BYTES, "image/png")},
        data={"family_id": family["id"], "vision_model": "m", "variants": "2"},
        auth=auth_headers,
    )

    assert response.status_code == 200
    assert len(response.json()["prompts"]) == 2
    assert mock_call.call_count == 2
    assert all(
        c.kwargs["image_data_uri"].startswith("data:image/png;base64,")
        for c in mock_call.call_args_list
    )


def test_from_image_rejects_too_many_variants(api_client, auth_headers, tmp_path):
    family = _create_family(tmp_path)

    response = api_client.post(
        "/api/from-image",
        files={"image": ("test.png", PNG_BYTES, "image/png")},
        data={"family_id": family["id"], "vision_model": "m", "variants": "5"},
        auth=auth_headers,
    )

    assert response.status_code == 422

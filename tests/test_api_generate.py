from unittest.mock import patch

from app import families


def _create_family(tmp_path, has_negative_prompt=True):
    return families.create_family(
        "SDXL", "family instructions", has_negative_prompt,
        path=str(tmp_path / "families.json"),
    )


def test_generate_requires_auth(api_client):
    response = api_client.post("/api/generate", json={
        "user_input": "a cat", "family_id": "x", "llm_model": "m",
    })

    assert response.status_code == 401


@patch("app.routes.api.call_openrouter")
def test_generate_returns_parsed_prompt(mock_call, api_client, auth_headers, tmp_path):
    family = _create_family(tmp_path)
    mock_call.return_value = "POSITIVE: a cyberpunk samurai\nNEGATIVE: blurry"

    response = api_client.post(
        "/api/generate",
        json={
            "user_input": "a cyberpunk samurai",
            "family_id": family["id"],
            "llm_model": "anthropic/claude-sonnet-4",
            "temperature": 0.7,
        },
        auth=auth_headers,
    )

    assert response.status_code == 200
    assert response.json() == {
        "positive_prompt": "a cyberpunk samurai",
        "negative_prompt": "blurry",
    }


@patch("app.routes.api.call_openrouter")
def test_generate_appends_to_history_as_generate_mode(mock_call, api_client, auth_headers, tmp_path):
    family = _create_family(tmp_path)
    mock_call.return_value = "POSITIVE: a cat\nNEGATIVE: blurry"

    api_client.post(
        "/api/generate",
        json={"user_input": "a cat", "family_id": family["id"], "llm_model": "m"},
        auth=auth_headers,
    )

    from app import history
    entries = history.list_entries(path=str(tmp_path / "history.jsonl"))
    assert len(entries) == 1
    assert entries[0]["mode"] == "generate"
    assert entries[0]["family_name"] == "SDXL"
    assert entries[0]["previous_prompt"] is None


def test_generate_rejects_blank_user_input(api_client, auth_headers, tmp_path):
    family = _create_family(tmp_path)

    response = api_client.post(
        "/api/generate",
        json={"user_input": "   ", "family_id": family["id"], "llm_model": "m"},
        auth=auth_headers,
    )

    assert response.status_code == 400


def test_generate_rejects_unknown_family(api_client, auth_headers):
    response = api_client.post(
        "/api/generate",
        json={"user_input": "a cat", "family_id": "nonexistent", "llm_model": "m"},
        auth=auth_headers,
    )

    assert response.status_code == 404


@patch("app.routes.api.call_openrouter")
def test_generate_returns_502_on_openrouter_error(mock_call, api_client, auth_headers, tmp_path):
    family = _create_family(tmp_path)
    mock_call.side_effect = RuntimeError("OpenRouter rate limit exceeded. Wait a moment and try again.")

    response = api_client.post(
        "/api/generate",
        json={"user_input": "a cat", "family_id": family["id"], "llm_model": "m"},
        auth=auth_headers,
    )

    assert response.status_code == 502
    assert "rate limit" in response.json()["detail"]


@patch("app.routes.api.call_openrouter")
def test_generate_with_previous_prompt_returns_parsed_prompt(mock_call, api_client, auth_headers, tmp_path):
    family = _create_family(tmp_path)
    mock_call.return_value = "POSITIVE: a samurai with lightning\nNEGATIVE: blurry"

    response = api_client.post(
        "/api/generate",
        json={
            "user_input": "add lightning",
            "previous_prompt": "a samurai in rain",
            "family_id": family["id"],
            "llm_model": "m",
        },
        auth=auth_headers,
    )

    assert response.status_code == 200
    assert response.json() == {
        "positive_prompt": "a samurai with lightning",
        "negative_prompt": "blurry",
    }


@patch("app.routes.api.call_openrouter")
def test_generate_with_previous_prompt_appends_to_history_as_iterate_mode(mock_call, api_client, auth_headers, tmp_path):
    family = _create_family(tmp_path)
    mock_call.return_value = "POSITIVE: updated\nNEGATIVE: "

    api_client.post(
        "/api/generate",
        json={
            "user_input": "add lightning", "previous_prompt": "a samurai",
            "family_id": family["id"], "llm_model": "m",
        },
        auth=auth_headers,
    )

    from app import history
    entries = history.list_entries(path=str(tmp_path / "history.jsonl"))
    assert entries[0]["mode"] == "iterate"
    assert entries[0]["previous_prompt"] == "a samurai"


@patch("app.routes.api.call_openrouter")
def test_generate_treats_blank_previous_prompt_as_generate_mode(mock_call, api_client, auth_headers, tmp_path):
    family = _create_family(tmp_path)
    mock_call.return_value = "POSITIVE: a cat\nNEGATIVE: "

    api_client.post(
        "/api/generate",
        json={
            "user_input": "a cat", "previous_prompt": "   ",
            "family_id": family["id"], "llm_model": "m",
        },
        auth=auth_headers,
    )

    from app import history
    entries = history.list_entries(path=str(tmp_path / "history.jsonl"))
    assert entries[0]["mode"] == "generate"
    assert entries[0]["previous_prompt"] is None


@patch("app.routes.api.call_openrouter")
def test_generate_with_previous_prompt_suppresses_negative_when_family_has_none(
    mock_call, api_client, auth_headers, tmp_path
):
    family = _create_family(tmp_path, has_negative_prompt=False)
    mock_call.return_value = "POSITIVE: updated flowing scene\nNEGATIVE: blurry"

    response = api_client.post(
        "/api/generate",
        json={
            "user_input": "make it night time", "previous_prompt": "a flowing scene",
            "family_id": family["id"], "llm_model": "m",
        },
        auth=auth_headers,
    )

    assert response.status_code == 200
    assert response.json() == {
        "positive_prompt": "updated flowing scene",
        "negative_prompt": "",
    }

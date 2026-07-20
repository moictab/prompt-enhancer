from unittest.mock import patch

from app import families


def _create_family(tmp_path, has_negative_prompt=True):
    return families.create_family(
        "SDXL", "family instructions", has_negative_prompt,
        path=str(tmp_path / "families.json"),
    )


def test_iterate_requires_auth(api_client):
    response = api_client.post("/api/iterate", json={
        "user_input": "add lightning", "previous_prompt": "a samurai",
        "family_id": "x", "llm_model": "m",
    })

    assert response.status_code == 401


@patch("app.routes.api.call_openrouter")
def test_iterate_returns_parsed_prompt(mock_call, api_client, auth_headers, tmp_path):
    family = _create_family(tmp_path)
    mock_call.return_value = "POSITIVE: a samurai with lightning\nNEGATIVE: blurry"

    response = api_client.post(
        "/api/iterate",
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


def test_iterate_rejects_blank_previous_prompt(api_client, auth_headers, tmp_path):
    family = _create_family(tmp_path)

    response = api_client.post(
        "/api/iterate",
        json={
            "user_input": "add lightning", "previous_prompt": "  ",
            "family_id": family["id"], "llm_model": "m",
        },
        auth=auth_headers,
    )

    assert response.status_code == 400


@patch("app.routes.api.call_openrouter")
def test_iterate_appends_to_history_with_previous_prompt(mock_call, api_client, auth_headers, tmp_path):
    family = _create_family(tmp_path)
    mock_call.return_value = "POSITIVE: updated\nNEGATIVE: "

    api_client.post(
        "/api/iterate",
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

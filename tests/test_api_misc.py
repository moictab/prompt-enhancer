from unittest.mock import patch

from app import characters, families, history


def test_get_history_requires_auth(api_client):
    response = api_client.get("/api/history")

    assert response.status_code == 401


def test_get_history_returns_entries_newest_first(api_client, auth_headers, tmp_path):
    path = str(tmp_path / "history.jsonl")
    history.append_entry(
        mode="generate", family_id="f", family_name="SDXL", llm_model="m",
        vision_model=None, temperature=0.7, user_input="first", example_prompts="",
        previous_prompt=None, positive_prompt="first result", negative_prompt="",
        path=path,
    )
    history.append_entry(
        mode="generate", family_id="f", family_name="SDXL", llm_model="m",
        vision_model=None, temperature=0.7, user_input="second", example_prompts="",
        previous_prompt=None, positive_prompt="second result", negative_prompt="",
        path=path,
    )

    response = api_client.get("/api/history", auth=auth_headers)

    assert response.status_code == 200
    body = response.json()
    assert [e["positive_prompt"] for e in body] == ["second result", "first result"]


def test_get_characters_requires_auth(api_client):
    response = api_client.get("/api/characters")

    assert response.status_code == 401


def test_get_characters_returns_list(api_client, auth_headers, tmp_path):
    characters.create_character("Warrior", "a fierce warrior", path=str(tmp_path / "characters.json"))

    response = api_client.get("/api/characters", auth=auth_headers)

    assert response.status_code == 200
    assert response.json()[0]["name"] == "Warrior"

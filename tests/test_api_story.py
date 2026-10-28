from unittest.mock import patch

from app import characters, families
from app.openrouter_client import OpenRouterResult


def _create_family(tmp_path, has_negative_prompt=True):
    return families.create_family(
        "SDXL", "family instructions", has_negative_prompt,
        path=str(tmp_path / "families.json"),
    )


STORY_RESPONSE = (
    "EPISODE 1:\nPOSITIVE: two friends sit down to dinner\nNEGATIVE: blurry\n\n"
    "EPISODE 2:\nPOSITIVE: the mood turns flirtatious\nNEGATIVE: blurry\n\n"
    "EPISODE 3:\nPOSITIVE: cards are dealt for strip poker\nNEGATIVE: blurry\n\n"
    "EPISODE 4:\nPOSITIVE: the game reaches its climax\nNEGATIVE: blurry"
)


def test_story_requires_auth(api_client):
    response = api_client.post("/api/story", json={
        "user_input": "two characters have dinner", "family_id": "x", "llm_model": "m",
    })

    assert response.status_code == 401


@patch("app.routes.api.call_openrouter")
def test_story_returns_parsed_episodes(mock_call, api_client, auth_headers, tmp_path):
    family = _create_family(tmp_path)
    mock_call.return_value = OpenRouterResult(content=STORY_RESPONSE, cost=0.0042)

    response = api_client.post(
        "/api/story",
        json={
            "user_input": "two characters have dinner and end up playing strip poker",
            "family_id": family["id"],
            "llm_model": "anthropic/claude-sonnet-4",
            "num_episodes": 4,
        },
        auth=auth_headers,
    )

    assert response.status_code == 200
    body = response.json()
    assert body["cost"] == 0.0042
    assert body["truncated"] is False
    assert body["prompts"] == [
        {"positive_prompt": "two friends sit down to dinner", "negative_prompt": "blurry"},
        {"positive_prompt": "the mood turns flirtatious", "negative_prompt": "blurry"},
        {"positive_prompt": "cards are dealt for strip poker", "negative_prompt": "blurry"},
        {"positive_prompt": "the game reaches its climax", "negative_prompt": "blurry"},
    ]
    assert body["count_mismatch"] is False


@patch("app.routes.api.call_openrouter")
def test_story_flags_episode_count_mismatch_when_model_ignores_format(mock_call, api_client, auth_headers, tmp_path):
    family = _create_family(tmp_path)
    mock_call.return_value = OpenRouterResult(
        content="POSITIVE: a single merged scene covering the whole story\nNEGATIVE: blurry", cost=0.0042
    )

    response = api_client.post(
        "/api/story",
        json={
            "user_input": "two characters have dinner",
            "family_id": family["id"],
            "llm_model": "m",
            "num_episodes": 4,
        },
        auth=auth_headers,
    )

    assert response.status_code == 200
    body = response.json()
    assert len(body["prompts"]) == 1
    assert body["count_mismatch"] is True


@patch("app.routes.api.call_openrouter")
def test_story_sends_requested_episode_count_to_llm(mock_call, api_client, auth_headers, tmp_path):
    family = _create_family(tmp_path)
    mock_call.return_value = OpenRouterResult(content=STORY_RESPONSE, cost=0.0042)

    api_client.post(
        "/api/story",
        json={
            "user_input": "two characters have dinner",
            "family_id": family["id"],
            "llm_model": "m",
            "num_episodes": 4,
        },
        auth=auth_headers,
    )

    sent_user_message = mock_call.call_args.kwargs["user_message"]
    assert "Generate exactly 4 episodes" in sent_user_message


@patch("app.routes.api.call_openrouter")
def test_story_suppresses_negative_when_family_has_none(mock_call, api_client, auth_headers, tmp_path):
    family = _create_family(tmp_path, has_negative_prompt=False)
    mock_call.return_value = OpenRouterResult(content=STORY_RESPONSE, cost=0.0042)

    response = api_client.post(
        "/api/story",
        json={
            "user_input": "two characters have dinner",
            "family_id": family["id"],
            "llm_model": "m",
            "num_episodes": 4,
        },
        auth=auth_headers,
    )

    assert all(ep["negative_prompt"] == "" for ep in response.json()["prompts"])


@patch("app.routes.api.call_openrouter")
def test_story_appends_a_single_grouped_history_entry(mock_call, api_client, auth_headers, tmp_path):
    family = _create_family(tmp_path)
    mock_call.return_value = OpenRouterResult(content=STORY_RESPONSE, cost=0.0042)

    api_client.post(
        "/api/story",
        json={
            "user_input": "two characters have dinner",
            "family_id": family["id"],
            "llm_model": "m",
            "num_episodes": 4,
        },
        auth=auth_headers,
    )

    from app import history
    entries = history.list_entries(path=str(tmp_path / "history.jsonl"))
    assert len(entries) == 1
    assert entries[0]["mode"] == "story"
    assert entries[0]["params"]["num_episodes"] == 4
    assert len(entries[0]["prompts"]) == 4


@patch("app.routes.api.call_openrouter")
def test_story_resolves_character_ids_into_user_message(mock_call, api_client, auth_headers, tmp_path):
    family = _create_family(tmp_path)
    character = characters.create_character(
        "Kaito", "a stoic ronin with a scarred left eye", path=str(tmp_path / "characters.json")
    )
    mock_call.return_value = OpenRouterResult(content=STORY_RESPONSE, cost=0.0042)

    api_client.post(
        "/api/story",
        json={
            "user_input": "two characters have dinner",
            "family_id": family["id"],
            "llm_model": "m",
            "num_episodes": 4,
            "character_ids": [character["id"]],
        },
        auth=auth_headers,
    )

    sent_user_message = mock_call.call_args.kwargs["user_message"]
    assert "## Characters" in sent_user_message
    assert "Kaito" in sent_user_message


def test_story_rejects_blank_user_input(api_client, auth_headers, tmp_path):
    family = _create_family(tmp_path)

    response = api_client.post(
        "/api/story",
        json={"user_input": "   ", "family_id": family["id"], "llm_model": "m"},
        auth=auth_headers,
    )

    assert response.status_code == 400


def test_story_rejects_unknown_family(api_client, auth_headers):
    response = api_client.post(
        "/api/story",
        json={"user_input": "a story", "family_id": "nonexistent", "llm_model": "m"},
        auth=auth_headers,
    )

    assert response.status_code == 404


def test_story_rejects_num_episodes_below_minimum(api_client, auth_headers, tmp_path):
    family = _create_family(tmp_path)

    response = api_client.post(
        "/api/story",
        json={"user_input": "a story", "family_id": family["id"], "llm_model": "m", "num_episodes": 1},
        auth=auth_headers,
    )

    assert response.status_code == 422


def test_story_rejects_num_episodes_above_maximum(api_client, auth_headers, tmp_path):
    family = _create_family(tmp_path)

    response = api_client.post(
        "/api/story",
        json={"user_input": "a story", "family_id": family["id"], "llm_model": "m", "num_episodes": 13},
        auth=auth_headers,
    )

    assert response.status_code == 422


@patch("app.routes.api.call_openrouter")
def test_story_returns_502_on_openrouter_error(mock_call, api_client, auth_headers, tmp_path):
    family = _create_family(tmp_path)
    mock_call.side_effect = RuntimeError("OpenRouter rate limit exceeded. Wait a moment and try again.")

    response = api_client.post(
        "/api/story",
        json={"user_input": "a story", "family_id": family["id"], "llm_model": "m"},
        auth=auth_headers,
    )

    assert response.status_code == 502
    assert "rate limit" in response.json()["detail"]

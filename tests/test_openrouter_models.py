from unittest.mock import Mock, patch

import pytest
import requests

from app import openrouter_client


@pytest.fixture(autouse=True)
def reset_models_cache():
    openrouter_client._models_cache["models"] = None
    openrouter_client._models_cache["fetched_at"] = 0.0
    yield
    openrouter_client._models_cache["models"] = None
    openrouter_client._models_cache["fetched_at"] = 0.0


def _mock_models_response(models):
    response = Mock()
    response.raise_for_status = Mock()
    response.json.return_value = {"data": models}
    return response


@patch("app.openrouter_client.requests.get")
def test_list_models_returns_simplified_id_name_list(mock_get):
    mock_get.return_value = _mock_models_response([
        {"id": "anthropic/claude-sonnet-4", "name": "Anthropic: Claude Sonnet 4", "extra": "ignored"},
        {"id": "openai/gpt-4o", "name": "OpenAI: GPT-4o"},
    ])

    result = openrouter_client.list_models()

    assert result == [
        {"id": "anthropic/claude-sonnet-4", "name": "Anthropic: Claude Sonnet 4"},
        {"id": "openai/gpt-4o", "name": "OpenAI: GPT-4o"},
    ]


@patch("app.openrouter_client.requests.get")
def test_list_models_falls_back_to_id_when_name_missing(mock_get):
    mock_get.return_value = _mock_models_response([{"id": "some/model"}])

    result = openrouter_client.list_models()

    assert result == [{"id": "some/model", "name": "some/model"}]


@patch("app.openrouter_client.requests.get")
def test_list_models_caches_and_does_not_refetch_within_ttl(mock_get):
    mock_get.return_value = _mock_models_response([{"id": "a/b", "name": "A B"}])

    openrouter_client.list_models()
    openrouter_client.list_models()

    assert mock_get.call_count == 1


@patch("app.openrouter_client.requests.get")
def test_list_models_refetches_after_ttl_expires(mock_get):
    mock_get.return_value = _mock_models_response([{"id": "a/b", "name": "A B"}])
    openrouter_client.list_models()
    openrouter_client._models_cache["fetched_at"] -= openrouter_client.MODELS_CACHE_TTL_SECONDS + 1

    openrouter_client.list_models()

    assert mock_get.call_count == 2


@patch("app.openrouter_client.requests.get")
def test_list_models_falls_back_to_stale_cache_on_error(mock_get):
    mock_get.return_value = _mock_models_response([{"id": "a/b", "name": "A B"}])
    openrouter_client.list_models()
    openrouter_client._models_cache["fetched_at"] -= openrouter_client.MODELS_CACHE_TTL_SECONDS + 1
    mock_get.side_effect = requests.exceptions.ConnectionError()

    result = openrouter_client.list_models()

    assert result == [{"id": "a/b", "name": "A B"}]


@patch("app.openrouter_client.requests.get")
def test_list_models_raises_when_no_cache_and_request_fails(mock_get):
    mock_get.side_effect = requests.exceptions.ConnectionError()

    with pytest.raises(RuntimeError, match="Could not fetch"):
        openrouter_client.list_models()

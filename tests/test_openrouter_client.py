from unittest.mock import Mock, patch

import pytest
import requests

from app.openrouter_client import call_openrouter


def _mock_response(status_code=200, json_data=None, text=""):
    response = Mock()
    response.status_code = status_code
    response.json.return_value = json_data or {}
    response.text = text
    return response


@patch("app.openrouter_client.requests.post")
def test_call_openrouter_returns_message_content(mock_post):
    mock_post.return_value = _mock_response(
        200, {"choices": [{"message": {"content": "POSITIVE: a cat\nNEGATIVE: blurry"}}]}
    )

    result = call_openrouter(
        api_key="key", model="anthropic/claude-sonnet-4",
        system_prompt="sys", user_message="user", temperature=0.7,
    )

    assert result.content == "POSITIVE: a cat\nNEGATIVE: blurry"


@patch("app.openrouter_client.requests.post")
def test_call_openrouter_returns_cost_from_usage(mock_post):
    mock_post.return_value = _mock_response(
        200, {
            "choices": [{"message": {"content": "ok"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "cost": 0.000035},
        }
    )

    result = call_openrouter(api_key="key", model="m", system_prompt="sys", user_message="user")

    assert result.cost == 0.000035


@patch("app.openrouter_client.requests.post")
def test_call_openrouter_returns_none_cost_when_usage_missing(mock_post):
    mock_post.return_value = _mock_response(200, {"choices": [{"message": {"content": "ok"}}]})

    result = call_openrouter(api_key="key", model="m", system_prompt="sys", user_message="user")

    assert result.cost is None


@patch("app.openrouter_client.requests.post")
def test_call_openrouter_requests_usage_accounting(mock_post):
    mock_post.return_value = _mock_response(200, {"choices": [{"message": {"content": "ok"}}]})

    call_openrouter(api_key="key", model="m", system_prompt="sys", user_message="user")

    sent_payload = mock_post.call_args.kwargs["json"]
    assert sent_payload["usage"] == {"include": True}


@patch("app.openrouter_client.requests.post")
def test_call_openrouter_sends_plain_string_content_without_image(mock_post):
    mock_post.return_value = _mock_response(200, {"choices": [{"message": {"content": "ok"}}]})

    call_openrouter(api_key="key", model="m", system_prompt="sys", user_message="hello")

    sent_payload = mock_post.call_args.kwargs["json"]
    assert sent_payload["messages"][1]["content"] == "hello"


@patch("app.openrouter_client.requests.post")
def test_call_openrouter_sends_multimodal_content_with_image(mock_post):
    mock_post.return_value = _mock_response(200, {"choices": [{"message": {"content": "ok"}}]})

    call_openrouter(
        api_key="key", model="m", system_prompt="sys", user_message="hello",
        image_data_uri="data:image/png;base64,AAAA",
    )

    sent_payload = mock_post.call_args.kwargs["json"]
    content = sent_payload["messages"][1]["content"]
    assert content == [
        {"type": "text", "text": "hello"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
    ]


@patch("app.openrouter_client.requests.post")
def test_call_openrouter_raises_on_timeout(mock_post):
    mock_post.side_effect = requests.exceptions.Timeout()

    with pytest.raises(RuntimeError, match="timed out"):
        call_openrouter(api_key="key", model="m", system_prompt="s", user_message="u")


@patch("app.openrouter_client.requests.post")
def test_call_openrouter_raises_on_401(mock_post):
    mock_post.return_value = _mock_response(401)

    with pytest.raises(RuntimeError, match="invalid"):
        call_openrouter(api_key="key", model="m", system_prompt="s", user_message="u")


@patch("app.openrouter_client.requests.post")
def test_call_openrouter_raises_on_402(mock_post):
    mock_post.return_value = _mock_response(402)

    with pytest.raises(RuntimeError, match="credits"):
        call_openrouter(api_key="key", model="m", system_prompt="s", user_message="u")


@patch("app.openrouter_client.requests.post")
def test_call_openrouter_raises_on_429(mock_post):
    mock_post.return_value = _mock_response(429)

    with pytest.raises(RuntimeError, match="rate limit"):
        call_openrouter(api_key="key", model="m", system_prompt="s", user_message="u")


@patch("app.openrouter_client.requests.post")
def test_call_openrouter_raises_on_malformed_response(mock_post):
    mock_post.return_value = _mock_response(200, {"unexpected": "shape"})

    with pytest.raises(RuntimeError, match="Unexpected response format"):
        call_openrouter(api_key="key", model="m", system_prompt="s", user_message="u")


@patch("app.openrouter_client.requests.post")
def test_call_openrouter_raises_on_null_content(mock_post):
    mock_post.return_value = _mock_response(200, {"choices": [{"message": {"content": None}}]})

    with pytest.raises(RuntimeError, match="Unexpected response format"):
        call_openrouter(api_key="key", model="m", system_prompt="s", user_message="u")


@patch("app.openrouter_client.requests.post")
def test_call_openrouter_raises_on_json_decode_error(mock_post):
    response = _mock_response(200, text="not json")
    response.json.side_effect = ValueError("not json")
    mock_post.return_value = response

    with pytest.raises(RuntimeError, match="Unexpected response format"):
        call_openrouter(api_key="key", model="m", system_prompt="s", user_message="u")

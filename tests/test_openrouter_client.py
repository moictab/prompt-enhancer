import json
from unittest.mock import MagicMock, Mock, patch

import pytest
import requests

from app.openrouter_client import StreamEnd, call_openrouter, stream_chat


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
def test_call_openrouter_tolerates_null_usage(mock_post):
    mock_post.return_value = _mock_response(
        200, {"choices": [{"message": {"content": "ok"}}], "usage": None}
    )

    result = call_openrouter(api_key="key", model="m", system_prompt="sys", user_message="user")

    assert result.content == "ok"
    assert result.cost is None


@patch("app.openrouter_client.requests.post")
def test_call_openrouter_treats_null_message_as_malformed_and_retries(mock_post):
    mock_post.side_effect = [
        _mock_response(200, {"choices": [{"message": None}]}),
        _mock_response(200, {"choices": [{"message": {"content": "ok"}}]}),
    ]

    result = call_openrouter(api_key="key", model="m", system_prompt="sys", user_message="user")

    assert result.content == "ok"
    assert mock_post.call_count == 2


@patch("app.openrouter_client.requests.post")
def test_call_openrouter_not_truncated_by_default(mock_post):
    mock_post.return_value = _mock_response(
        200, {"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]}
    )

    result = call_openrouter(api_key="key", model="m", system_prompt="sys", user_message="user")

    assert result.truncated is False


@patch("app.openrouter_client.requests.post")
def test_call_openrouter_flags_truncated_on_length_finish_reason(mock_post):
    mock_post.return_value = _mock_response(
        200, {"choices": [{"message": {"content": "ok"}, "finish_reason": "length"}]}
    )

    result = call_openrouter(api_key="key", model="m", system_prompt="sys", user_message="user")

    assert result.truncated is True


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

    assert mock_post.call_count == 2


@patch("app.openrouter_client.requests.post")
def test_call_openrouter_retries_once_and_succeeds(mock_post):
    mock_post.side_effect = [
        _mock_response(200, {"choices": [{"message": {"content": None}}]}, text="   \n   "),
        _mock_response(200, {"choices": [{"message": {"content": "POSITIVE: a cat"}}]}),
    ]

    result = call_openrouter(api_key="key", model="m", system_prompt="s", user_message="u")

    assert result.content == "POSITIVE: a cat"
    assert mock_post.call_count == 2


@patch("app.openrouter_client.requests.post")
def test_call_openrouter_raises_on_json_decode_error(mock_post):
    response = _mock_response(200, text="not json")
    response.json.side_effect = ValueError("not json")
    mock_post.return_value = response

    with pytest.raises(RuntimeError, match="Unexpected response format"):
        call_openrouter(api_key="key", model="m", system_prompt="s", user_message="u")


def _stream_response(lines, status_code=200, text=""):
    response = MagicMock()
    response.status_code = status_code
    response.text = text
    response.iter_lines.return_value = [line.encode("utf-8") for line in lines]
    response.__enter__.return_value = response
    return response


def _data(payload):
    return "data: " + json.dumps(payload)


@patch("app.openrouter_client.requests.post")
def test_stream_chat_yields_deltas_then_the_end_with_cost(mock_post):
    mock_post.return_value = _stream_response([
        ": OPENROUTER PROCESSING",
        _data({"choices": [{"delta": {"role": "assistant", "content": ""}}]}),
        _data({"choices": [{"delta": {"content": "Hola, "}}]}),
        "",
        _data({"choices": [{"delta": {"content": "Lena."}, "finish_reason": "stop"}]}),
        _data({"choices": [], "usage": {"cost": 0.0042}}),
        "data: [DONE]",
    ])

    items = list(stream_chat("key", "m", [{"role": "user", "content": "hi"}], 0.5))

    assert items[:-1] == ["Hola, ", "Lena."]
    assert items[-1] == StreamEnd(cost=0.0042, truncated=False)
    payload = mock_post.call_args.kwargs["json"]
    assert payload["stream"] is True
    assert payload["messages"] == [{"role": "user", "content": "hi"}]
    assert payload["temperature"] == 0.5
    assert mock_post.call_args.kwargs["stream"] is True


@patch("app.openrouter_client.requests.post")
def test_stream_chat_flags_a_reply_cut_by_the_length_limit(mock_post):
    mock_post.return_value = _stream_response([
        _data({"choices": [{"delta": {"content": "long"}, "finish_reason": "length"}]}),
        "data: [DONE]",
    ])

    items = list(stream_chat("key", "m", [], 0.7))

    assert items[-1].truncated is True


@patch("app.openrouter_client.requests.post")
def test_stream_chat_raises_on_an_error_reported_mid_stream(mock_post):
    mock_post.return_value = _stream_response([
        _data({"choices": [{"delta": {"content": "Empie"}}]}),
        _data({"error": {"message": "Provider overloaded"}}),
    ])

    stream = stream_chat("key", "m", [], 0.7)
    assert next(stream) == "Empie"
    with pytest.raises(RuntimeError, match="Provider overloaded"):
        next(stream)


@patch("app.openrouter_client.requests.post")
def test_stream_chat_raises_on_http_errors(mock_post):
    mock_post.return_value = _stream_response([], status_code=402)

    with pytest.raises(RuntimeError, match="insufficient credits"):
        list(stream_chat("key", "m", [], 0.7))


@patch("app.openrouter_client.requests.post")
def test_stream_chat_raises_when_the_connection_drops(mock_post):
    response = _stream_response([])
    response.iter_lines.side_effect = requests.exceptions.ChunkedEncodingError("reset")
    mock_post.return_value = response

    with pytest.raises(RuntimeError, match="dropped mid-reply"):
        list(stream_chat("key", "m", [], 0.7))

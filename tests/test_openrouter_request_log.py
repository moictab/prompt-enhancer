"""The request log as filled by the OpenRouter client: what was really sent and what came back."""
import json
from unittest.mock import MagicMock, patch

import pytest

from app import request_log
from app.openrouter_client import OpenRouterError, stream_chat


def _stream_response(lines, status_code=200, text=""):
    response = MagicMock()
    response.status_code = status_code
    response.text = text
    response.iter_lines.return_value = [line.encode("utf-8") for line in lines]
    response.__enter__.return_value = response
    return response


def _data(payload):
    return "data: " + json.dumps(payload)


def _entries():
    return request_log.list_summaries(limit=200)["items"]


@patch("app.openrouter_client.requests.post")
def test_stream_logs_the_text_as_it_arrived_with_usage(mock_post):
    mock_post.return_value = _stream_response([
        _data({"id": "gen-9", "provider": "Anthropic", "choices": [{"delta": {"content": "Hola, "}}]}),
        _data({"choices": [{"delta": {"content": "Lena."}, "finish_reason": "stop"}]}),
        _data({"choices": [], "usage": {"cost": 0.004, "completion_tokens": 3}}),
        "data: [DONE]",
    ])
    messages = [{"role": "system", "content": "persona"}, {"role": "user", "content": "hola"}]

    items = list(stream_chat("k", "m", messages, 0.5, context={"mode": "conversation", "conversation_id": "c1"}))

    entry = request_log.get(items[-1].request_id)
    assert entry["request"]["messages"] == messages
    assert entry["request"]["stream"] is True
    assert entry["response"]["content"] == "Hola, Lena."
    assert entry["response"]["finish_reason"] == "stop"
    assert entry["response"]["usage"] == {"cost": 0.004, "completion_tokens": 3}
    assert entry["response"]["id"] == "gen-9"
    assert entry["response"]["provider"] == "Anthropic"
    assert entry["mode"] == "conversation"
    assert entry["context"] == {"conversation_id": "c1"}
    assert entry["error"] is None


@patch("app.openrouter_client.requests.post")
def test_stream_logs_what_arrived_before_an_error(mock_post):
    mock_post.return_value = _stream_response([
        _data({"choices": [{"delta": {"content": "Empie"}}]}),
        _data({"error": {"message": "Provider overloaded"}}),
    ])

    with pytest.raises(OpenRouterError) as raised:
        list(stream_chat("k", "m", [], 0.7))

    entry = request_log.get(raised.value.request_id)
    assert entry["response"]["content"] == "Empie"
    assert "Provider overloaded" in entry["error"]


@patch("app.openrouter_client.requests.post")
def test_stream_logs_an_http_error_with_its_body(mock_post):
    mock_post.return_value = _stream_response([], status_code=404, text="model gone")

    with pytest.raises(OpenRouterError) as raised:
        list(stream_chat("k", "m", [], 0.7))

    entry = request_log.get(raised.value.request_id)
    assert entry["status"] == 404
    assert entry["response_body"] == "model gone"


@patch("app.openrouter_client.requests.post")
def test_stream_abandoned_by_the_caller_is_logged_as_such(mock_post):
    mock_post.return_value = _stream_response([
        _data({"choices": [{"delta": {"content": "Primera "}}]}),
        _data({"choices": [{"delta": {"content": "parte"}}]}),
    ])

    stream = stream_chat("k", "m", [], 0.7)
    assert next(stream) == "Primera "
    stream.close()

    [summary] = _entries()
    entry = request_log.get(summary["id"])
    assert entry["error"].startswith("Abandoned")
    assert entry["response"]["content"] == "Primera "

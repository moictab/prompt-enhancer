"""The request log as filled by the OpenRouter client: what was really sent and what came back."""
import json
from unittest.mock import MagicMock, Mock, patch

import pytest

from app import request_log
from app.openrouter_client import OpenRouterError, call_openrouter, stream_chat


def _response(status_code=200, json_data=None, text=""):
    response = Mock()
    response.status_code = status_code
    response.json.return_value = json_data or {}
    response.text = text
    return response


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
def test_call_logs_the_body_it_posted_and_the_full_response(mock_post):
    body = {
        "id": "gen-1",
        "choices": [{"message": {"content": "POSITIVE: a cat"}, "finish_reason": "stop"}],
        "usage": {"cost": 0.002, "prompt_tokens": 50},
    }
    mock_post.return_value = _response(200, body)

    result = call_openrouter(
        api_key="secret-key", model="m", system_prompt="sys", user_message="user",
        temperature=0.3, context={"mode": "generate"},
    )

    entry = request_log.get(result.request_id)
    assert entry["request"] == mock_post.call_args.kwargs["json"]
    assert entry["response"] == body
    assert entry["mode"] == "generate"
    assert entry["status"] == 200
    assert "secret-key" not in json.dumps(entry)


@patch("app.openrouter_client.requests.post")
def test_call_logs_the_image_once_outside_the_payload(mock_post):
    mock_post.return_value = _response(200, {"choices": [{"message": {"content": "ok"}}]})

    result = call_openrouter(
        api_key="k", model="m", system_prompt="s", user_message="u",
        image_data_uri="data:image/png;base64,iVBORw0KGgo=",
    )

    content = request_log.get(result.request_id)["request"]["messages"][1]["content"]
    assert content[1]["image_url"]["url"].startswith(request_log.IMAGE_MARKER)
    # The body actually posted still carried the image itself.
    assert mock_post.call_args.kwargs["json"]["messages"][1]["content"][1]["image_url"]["url"].startswith("data:")


@patch("app.openrouter_client.requests.post")
def test_call_logs_an_http_error_with_its_body(mock_post):
    mock_post.return_value = _response(404, text='{"error":{"message":"model gone"}}')

    with pytest.raises(OpenRouterError) as raised:
        call_openrouter(api_key="k", model="m", system_prompt="s", user_message="u", context={"mode": "story"})

    entry = request_log.get(raised.value.request_id)
    assert entry["status"] == 404
    assert "HTTP 404" in entry["error"]
    assert entry["response_body"] == '{"error":{"message":"model gone"}}'


@patch("app.openrouter_client.requests.post")
def test_call_logs_each_attempt_of_a_retry(mock_post):
    mock_post.side_effect = [
        _response(200, {"choices": [{"message": None}]}, text="{}"),
        _response(200, {"choices": [{"message": {"content": "ok"}}]}),
    ]

    result = call_openrouter(api_key="k", model="m", system_prompt="s", user_message="u")

    entries = _entries()
    assert [e["attempt"] for e in entries] == [2, 1]
    assert entries[0]["id"] == result.request_id
    assert entries[1]["error"].startswith("Unexpected response format")


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

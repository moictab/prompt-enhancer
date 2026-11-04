import json
from unittest.mock import MagicMock, patch

import pytest
import requests

from app import openrouter_client, request_log
from app.openrouter_client import stream_chat


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
    assert items[-1].cost == 0.0042
    assert items[-1].truncated is False
    assert items[-1].request_id
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


@patch("app.openrouter_client.requests.post")
def test_stream_chat_assembles_tool_calls_and_sends_the_tools(mock_post):
    tools = [{"type": "function", "function": {"name": "run_workflow", "parameters": {"type": "object"}}}]
    mock_post.return_value = _stream_response([
        _data({"choices": [{"delta": {"content": "Voy."}}]}),
        _data({"choices": [{"delta": {"tool_calls": [
            {"index": 0, "id": "call_1", "type": "function", "function": {"name": "run_workflow", "arguments": ""}}]}}]}),
        _data({"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": '{"workflow": '}}]}}]}),
        _data({"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": '"w"}'}}]}}]}),
        _data({"choices": [{"delta": {"tool_calls": [
            {"index": 1, "id": "call_2", "function": {"name": "inspect_workflow", "arguments": "{}"}}]}}]}),
        _data({"choices": [{"delta": {}, "finish_reason": "tool_calls"}]}),
        "data: [DONE]",
    ])

    items = list(openrouter_client.stream_chat("k", "m", [{"role": "user", "content": "x"}],
                                               tools=tools, tool_choice="none"))

    assert items[0] == "Voy."
    end = items[-1]
    assert end.tool_calls == [
        {"id": "call_1", "name": "run_workflow", "arguments": '{"workflow": "w"}'},
        {"id": "call_2", "name": "inspect_workflow", "arguments": "{}"},
    ]
    payload = mock_post.call_args.kwargs["json"]
    assert payload["tools"] == tools and payload["tool_choice"] == "none"
    logged = request_log.get(end.request_id)
    assert logged["response"]["tool_calls"] == end.tool_calls


@patch("app.openrouter_client.requests.post")
def test_stream_chat_sends_no_tools_unless_given(mock_post):
    mock_post.return_value = _stream_response([_data({"choices": [{"delta": {"content": "hi"}}]}), "data: [DONE]"])

    list(openrouter_client.stream_chat("k", "m", []))

    assert "tools" not in mock_post.call_args.kwargs["json"]


@patch("app.openrouter_client.requests.post")
def test_every_request_asks_for_providers_that_keep_nothing(mock_post):
    mock_post.return_value = _stream_response([_data({"choices": [{"delta": {"content": "hi"}}]}), "data: [DONE]"])
    list(openrouter_client.stream_chat("k", "m", []))
    assert mock_post.call_args.kwargs["json"]["provider"] == {"data_collection": "deny"}


@patch("app.openrouter_client.requests.post")
def test_no_provider_meeting_the_privacy_requirement_is_explained(mock_post):
    mock_post.return_value = _stream_response(
        [], status_code=404,
        text='{"error":{"message":"No endpoints found matching your data policy (Zero data retention)"}}')

    with pytest.raises(RuntimeError, match="privacy requirement"):
        list(openrouter_client.stream_chat("k", "m", []))

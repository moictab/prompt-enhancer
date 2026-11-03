from unittest.mock import MagicMock, patch

import pytest
import requests

from app import comfy


def _response(status=200, body=None, text="", headers=None, content=b""):
    response = MagicMock()
    response.status_code = status
    response.json.return_value = body
    response.text = text
    response.headers = headers or {}
    response.content = content
    return response


def test_client_is_off_without_urls_and_built_from_settings(monkeypatch):
    assert comfy.client() is None

    monkeypatch.setenv("COMFY_GATEWAY_URL", "http://gw:8189/")
    monkeypatch.setenv("COMFY_DIRECT_URL", "http://comfy:8188")
    client = comfy.client()

    assert (client.gateway_url, client.direct_url) == ("http://gw:8189", "http://comfy:8188")
    assert comfy.client() is client


@patch("app.comfy.requests.request")
def test_pick_base_falls_back_to_comfyui_only_when_the_gateway_doesnt_answer(mock_request):
    client = comfy.Comfy("http://gw", "http://comfy")
    mock_request.side_effect = [requests.exceptions.ConnectionError(), _response(body={})]

    assert client.pick_base() == "http://comfy"
    assert [c.args[1] for c in mock_request.call_args_list] == ["http://gw/models", "http://comfy/models"]

    # A 500 from the gateway is an answer: it stays.
    client = comfy.Comfy("http://gw", "http://comfy")
    mock_request.side_effect = [_response(status=500, body={"error": "boom"})]
    assert client.pick_base() == "http://gw"


@patch("app.comfy.requests.request")
def test_pick_base_reports_both_down(mock_request):
    mock_request.side_effect = requests.exceptions.ConnectionError()

    with pytest.raises(comfy.Unreachable):
        comfy.Comfy("http://gw", "http://comfy").pick_base()


@patch("app.comfy.requests.request")
def test_submit_declares_project_priority_and_tag(mock_request):
    mock_request.return_value = _response(body={"prompt_id": "abc", "number": 3})
    client = comfy.Comfy("http://gw")

    assert client.submit("http://gw", {"1": {"class_type": "X", "inputs": {}}}) == "abc"

    body = mock_request.call_args.kwargs["json"]
    assert body["extra_data"] == {"gateway": {"project": "prompt-enhancer", "priority": "high", "tag": "conversar"}}
    assert body["client_id"] == client.client_id


@patch("app.comfy.requests.request")
def test_validation_errors_carry_node_errors(mock_request):
    mock_request.return_value = _response(status=400, body={
        "error": {"type": "prompt_outputs_failed_validation", "message": "Prompt outputs failed validation"},
        "node_errors": {"141": {"errors": [{"message": "Value not in list", "details": "lora_name: 'x'"}],
                                "class_type": "LoraLoader"}},
    })

    with pytest.raises(comfy.ComfyError) as error:
        comfy.Comfy("http://gw").submit("http://gw", {})

    assert str(error.value) == "Prompt outputs failed validation"
    assert error.value.node_errors["141"]["class_type"] == "LoraLoader"


def test_result_from_history():
    assert comfy.result_from_history({}) is None
    assert comfy.result_from_history({"status": {"status_str": "running", "completed": False}}) is None

    done = comfy.result_from_history({
        "status": {"status_str": "success", "completed": True},
        "outputs": {"9": {"images": [{"filename": "a.png", "subfolder": "x", "type": "output"},
                                     ]},
                    "12": {"images": [{"filename": "p.png", "type": "temp"}]}},
    })
    assert done.status == "success"
    assert done.outputs == [{"node": "9", "filename": "a.png", "subfolder": "x", "type": "output"}]

    failed = comfy.result_from_history({"status": {"status_str": "error", "messages": [
        ["execution_error", {"exception_type": "OOM", "exception_message": "out of memory", "node_type": "KSampler"}],
    ]}})
    assert (failed.status, failed.error) == ("error", "OOM: out of memory (in KSampler)")

    interrupted = comfy.result_from_history({"status": {"status_str": "error", "messages": [
        ["execution_interrupted", {}]]}})
    assert interrupted.status == "interrupted"


@patch("app.comfy.requests.request")
def test_models_are_cached(mock_request):
    mock_request.side_effect = [_response(body={}), _response(body=["a.safetensors", 3, "b.safetensors"])]
    client = comfy.Comfy("http://gw")

    assert client.models("loras") == ["a.safetensors", "b.safetensors"]
    assert client.models("loras") == ["a.safetensors", "b.safetensors"]
    assert mock_request.call_count == 2  # system_stats + models, once

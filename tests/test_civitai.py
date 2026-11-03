import json
from unittest.mock import MagicMock, patch

import pytest

from app import civitai, conversations, families, image_metadata
from app.openrouter_client import StreamEnd

from .conftest import png_bytes

GRAPH = {"3": {"class_type": "KSampler", "inputs": {"seed": 1, "positive": ["6", 0]}},
         "6": {"class_type": "CLIPTextEncode", "inputs": {"text": "a fox"}}}
IMAGE_URL = "https://image.civitai.com/key/uuid/original=true/123"

ITEM = {
    "id": 123, "type": "image", "url": IMAGE_URL, "width": 832, "height": 1216, "username": "someone",
    "baseModel": "Krea 2",
    "meta": {
        "prompt": "masterpiece, a witch at dawn", "negativePrompt": "blurry", "seed": 42, "steps": 8,
        "cfgScale": 1, "sampler": "euler",
        "civitaiResources": [{"type": "lora", "weight": 0.8, "modelVersionId": 5},
                             {"type": "checkpoint", "weight": 1, "modelVersionId": 6}],
        "resources": [{"name": "old-style", "type": "lora", "weight": 0.5, "hash": "abc123"}],
        "additionalResources": [{"name": "Moss/RealismEnhancer.safetensors", "type": "lora", "strength": 1}],
        "comfy": json.dumps({"prompt": GRAPH, "workflow": {"nodes": [], "links": []}}),
    },
}
VERSIONS = {
    "/model-versions/5": {"name": "v5.1 [krea2]", "baseModel": "Krea 2", "trainedWords": ["masterpiece", "very aesthetic"],
                          "model": {"name": "Aesthetic Quality Modifiers", "type": "LORA"},
                          "files": [{"name": "krea2-masterpieces-v51.safetensors", "primary": True}]},
    "/model-versions/6": {"name": "v1", "baseModel": "Krea 2", "model": {"name": "Krea 2 Turbo", "type": "Checkpoint"},
                          "files": [{"name": "krea2-turbo.safetensors"}]},
}


def _response(status=200, body=None, content=b"", content_type="application/json"):
    response = MagicMock()
    response.status_code = status
    response.json.return_value = body
    response.content = content
    response.headers = {"Content-Type": content_type}
    return response


def _fake_get(item=ITEM, image=None, image_type="image/png"):
    calls = []

    def get(url, params=None, headers=None, timeout=None):
        calls.append({"url": url, "params": params, "headers": headers})
        if url == IMAGE_URL:
            return _response(content=image or png_bytes(), content_type=image_type)
        path = url.removeprefix(civitai.API)
        if path == "/images":
            return _response(body={"items": [item] if item else []})
        if path in VERSIONS:
            return _response(body=VERSIONS[path])
        return _response(status=404)

    return get, calls


@pytest.fixture(autouse=True)
def fresh_cache():
    civitai._versions.clear()


def test_image_ids_finds_civitai_image_links():
    text = "mira https://civitai.com/images/123?postId=4 y https://civitai.green/images/77, otra vez civitai.com/images/9 " \
           "y de nuevo https://www.civitai.com/images/123"
    assert civitai.image_ids(text) == [123, 77]


def test_fetch_reads_the_image_its_generation_data_and_resources(monkeypatch):
    monkeypatch.setenv("CIVITAI_API_KEY", "secret-key")
    get, calls = _fake_get()

    with patch("app.civitai.requests.get", get):
        shared = civitai.fetch_image(123)

    assert shared.content_type == "image/png"
    assert shared.workflow == GRAPH
    m = shared.metadata
    assert m["generator"] == "Civitai" and m["has_workflow"] is True and m["format"] == "PNG"
    assert m["source"] == {"site": "Civitai", "image_id": 123, "url": "https://civitai.com/images/123",
                           "author": "someone"}
    assert m["positive"] == "masterpiece, a witch at dawn" and m["negative"] == "blurry"
    assert m["settings"] == {"seed": 42, "steps": 8, "cfg": 1, "sampler": "euler", "base model": "Krea 2"}
    assert m["resources"] == [
        {"type": "lora", "name": "Aesthetic Quality Modifiers", "version": "v5.1 [krea2]", "base_model": "Krea 2",
         "trigger_words": ["masterpiece", "very aesthetic"], "file": "krea2-masterpieces-v51.safetensors",
         "weight": 0.8},
        {"type": "checkpoint", "name": "Krea 2 Turbo", "version": "v1", "base_model": "Krea 2",
         "file": "krea2-turbo.safetensors"},
        {"type": "lora", "name": "old-style", "weight": 0.5},  # its hash isn't on Civitai
        {"type": "lora", "name": "Moss/RealismEnhancer.safetensors", "weight": 1},
    ]
    api_call = calls[0]
    assert api_call["params"] == {"imageId": 123, "withMeta": "true", "nsfw": "X"}
    assert api_call["headers"]["Authorization"] == "Bearer secret-key"
    # The key goes to the API only, never to the image host.
    assert "Authorization" not in next(c for c in calls if c["url"] == IMAGE_URL)["headers"]


def test_fetch_reads_the_meta_nested_as_the_api_returns_it_by_image_id():
    get, _ = _fake_get(item={**ITEM, "meta": {"id": 123, "meta": ITEM["meta"]}})

    with patch("app.civitai.requests.get", get):
        shared = civitai.fetch_image(123)

    assert shared.metadata["positive"] == "masterpiece, a witch at dawn"
    assert len(shared.metadata["resources"]) == 4
    assert shared.workflow == GRAPH


def test_fetch_failures_are_explained():
    for item, image_type, expected in (
        (None, "image/png", "has no image 123"),
        ({**ITEM, "type": "video"}, "image/png", "is a video"),
        (ITEM, "text/html", "Couldn't download"),
    ):
        get, _ = _fake_get(item=item, image_type=image_type)
        with patch("app.civitai.requests.get", get), pytest.raises(civitai.CivitaiError, match=expected):
            civitai.fetch_image(123)


def test_describe_tells_the_llm_where_it_comes_from_and_what_made_it():
    get, _ = _fake_get()
    with patch("app.civitai.requests.get", get):
        metadata = civitai.fetch_image(123).metadata

    lines = image_metadata.describe(metadata, 2).splitlines()

    assert lines[0] == ('Image 2 metadata (PNG, 832x1216, made with Civitai; its ComfyUI workflow can be used as '
                        '"image:2"):')
    assert lines[1] == "- Source: Civitai image 123, by someone"
    assert lines[4] == (
        '- Civitai resources: lora "Aesthetic Quality Modifiers" v5.1 [krea2] (base Krea 2; weight 0.8; trigger words: '
        'masterpiece, very aesthetic; file krea2-masterpieces-v51.safetensors); checkpoint "Krea 2 Turbo" v1 '
        '(base Krea 2; file krea2-turbo.safetensors); lora "old-style" (weight 0.5); '
        'lora "Moss/RealismEnhancer.safetensors" (weight 1)'
    )


def _conversation(api_client, auth_headers, tmp_path):
    family = families.create_family("Krea", "rules", False, path=str(tmp_path / "families.json"))
    return api_client.post("/api/conversations", auth=auth_headers,
                           json={"family_id": family["id"], "llm_model": "m"}).json()["id"]


def test_a_civitai_link_in_a_message_brings_the_image_in(api_client, auth_headers, tmp_path):
    conversation_id = _conversation(api_client, auth_headers, tmp_path)
    get, _ = _fake_get()
    sent = []

    def reply(api_key, model, messages, temperature, context=None, tools=None, tool_choice=None):
        sent.append(messages)
        yield "Qué estilo."
        yield StreamEnd(cost=None)

    with patch("app.civitai.requests.get", get), patch("app.conversation_turn.stream_chat", reply):
        response = api_client.post(f"/api/conversations/{conversation_id}/messages", auth=auth_headers,
                                   data={"content": "Inspírate en https://civitai.com/images/123"})

    assert response.status_code == 200
    [image] = conversations.get(conversation_id)["messages"][0]["images"]
    assert image["metadata"]["source"]["image_id"] == 123
    assert conversations.image_workflow(conversation_id, image["id"]) == GRAPH
    texts = [p["text"] for p in sent[0][-1]["content"] if p["type"] == "text"]
    assert texts[0] == "Inspírate en https://civitai.com/images/123"
    assert "- Source: Civitai image 123, by someone" in texts[1]


def test_a_civitai_link_that_fails_stores_nothing(api_client, auth_headers, tmp_path):
    conversation_id = _conversation(api_client, auth_headers, tmp_path)
    get, _ = _fake_get(item=None)

    with patch("app.civitai.requests.get", get):
        response = api_client.post(f"/api/conversations/{conversation_id}/messages", auth=auth_headers,
                                   data={"content": "https://civitai.com/images/123"})

    assert response.status_code == 400
    assert "has no image 123" in response.json()["detail"]
    assert conversations.get(conversation_id)["messages"] == []


def test_civitai_links_count_towards_the_image_limit(api_client, auth_headers, tmp_path):
    conversation_id = _conversation(api_client, auth_headers, tmp_path)
    files = [("images", (f"{i}", png_bytes(), "image/png")) for i in range(4)]

    response = api_client.post(f"/api/conversations/{conversation_id}/messages", auth=auth_headers,
                               data={"content": "y https://civitai.com/images/123"}, files=files)

    assert response.status_code == 400
    assert "Civitai links included" in response.json()["detail"]

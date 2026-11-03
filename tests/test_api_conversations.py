import json
from unittest.mock import patch

from app import characters, families, history, memory, system_prompt
from app.openrouter_client import StreamEnd

REPLY = (
    "<title>Lena en la playa</title>\nMi propuesta:\n"
    "<prompt>\nPOSITIVE: Lena on a beach at dusk\nNEGATIVE: blurry\n</prompt>\n"
    "<remember>Likes golden hour</remember>\n¿Más cálida?"
)


def _family(tmp_path, has_negative_prompt=True):
    return families.create_family(
        "SDXL", "family instructions", has_negative_prompt, path=str(tmp_path / "families.json")
    )


def _stream(*chunks, cost=0.003, truncated=False):
    def fake(api_key, model, messages, temperature):
        yield from chunks
        yield StreamEnd(cost=cost, truncated=truncated)
    return fake


def _events(response):
    return [json.loads(line) for line in response.text.splitlines() if line.strip()]


def _create(api_client, auth_headers, family, **overrides):
    payload = {"family_id": family["id"], "llm_model": "vendor/model", "temperature": 0.5}
    payload.update(overrides)
    return api_client.post("/api/conversations", json=payload, auth=auth_headers)


def _send(api_client, auth_headers, conversation_id, content="Lena en la playa", files=None):
    return api_client.post(
        f"/api/conversations/{conversation_id}/messages",
        data={"content": content},
        files=files,
        auth=auth_headers,
    )


def test_conversations_require_auth(api_client):
    assert api_client.get("/api/conversations").status_code == 401


def test_create_get_list_and_delete_a_conversation(api_client, auth_headers, tmp_path):
    family = _family(tmp_path)

    created = _create(api_client, auth_headers, family).json()

    assert created["family_id"] == family["id"]
    assert created["llm_model"] == "vendor/model"
    assert api_client.get(f"/api/conversations/{created['id']}", auth=auth_headers).json() == created
    assert [c["id"] for c in api_client.get("/api/conversations", auth=auth_headers).json()] == [created["id"]]

    assert api_client.delete(f"/api/conversations/{created['id']}", auth=auth_headers).status_code == 200
    assert api_client.get(f"/api/conversations/{created['id']}", auth=auth_headers).status_code == 404


def test_create_rejects_unknown_family_and_blank_model(api_client, auth_headers, tmp_path):
    family = _family(tmp_path)

    assert _create(api_client, auth_headers, family, family_id="nope").status_code == 404
    assert _create(api_client, auth_headers, family, llm_model="  ").status_code == 400


def test_update_changes_the_settings(api_client, auth_headers, tmp_path):
    family = _family(tmp_path)
    conversation = _create(api_client, auth_headers, family).json()

    response = api_client.patch(
        f"/api/conversations/{conversation['id']}",
        json={"llm_model": "other/model", "temperature": 0.9},
        auth=auth_headers,
    )

    assert response.status_code == 200
    assert response.json()["llm_model"] == "other/model"
    assert response.json()["temperature"] == 0.9
    assert response.json()["family_id"] == family["id"]


def test_update_rejects_unknown_family(api_client, auth_headers, tmp_path):
    family = _family(tmp_path)
    conversation = _create(api_client, auth_headers, family).json()

    response = api_client.patch(
        f"/api/conversations/{conversation['id']}", json={"family_id": "nope"}, auth=auth_headers
    )

    assert response.status_code == 404


@patch("app.routes.conversations.stream_chat")
def test_send_message_streams_the_reply_and_stores_it(mock_stream, api_client, auth_headers, tmp_path):
    family = _family(tmp_path)
    conversation = _create(api_client, auth_headers, family).json()
    mock_stream.side_effect = _stream(REPLY[:40], REPLY[40:])

    response = _send(api_client, auth_headers, conversation["id"])

    assert response.status_code == 200
    events = _events(response)
    assert [e["type"] for e in events] == ["delta", "delta", "done"]
    assert "".join(e["text"] for e in events[:2]) == REPLY

    done = events[-1]
    message = done["message"]
    assert message["role"] == "assistant"
    assert message["parts"] == [
        {"type": "text", "text": "Mi propuesta:"},
        {"type": "prompt", "positive_prompt": "Lena on a beach at dusk", "negative_prompt": "blurry"},
        {"type": "text", "text": "¿Más cálida?"},
    ]
    assert message["cost"] == 0.003
    assert done["conversation"]["title"] == "Lena en la playa"
    assert done["conversation"]["cost"] == 0.003

    stored = api_client.get(f"/api/conversations/{conversation['id']}", auth=auth_headers).json()
    assert [m["role"] for m in stored["messages"]] == ["user", "assistant"]
    assert stored["messages"][0]["content"] == "Lena en la playa"
    assert stored["title"] == "Lena en la playa"


@patch("app.routes.conversations.stream_chat")
def test_reply_notes_go_to_memory(mock_stream, api_client, auth_headers, tmp_path):
    family = _family(tmp_path)
    conversation = _create(api_client, auth_headers, family).json()
    mock_stream.side_effect = _stream(REPLY)

    message = _events(_send(api_client, auth_headers, conversation["id"]))[-1]["message"]

    [note] = memory.list_notes(path=str(tmp_path / "memory.json"))
    assert note["text"] == "Likes golden hour"
    assert note["conversation_id"] == conversation["id"]
    assert message["notes"] == [{"id": note["id"], "text": "Likes golden hour"}]


@patch("app.routes.conversations.stream_chat")
def test_reply_with_prompts_is_logged_to_history(mock_stream, api_client, auth_headers, tmp_path):
    family = _family(tmp_path)
    character = characters.create_character("Lena", "auburn hair", path=str(tmp_path / "characters.json"))
    conversation = _create(api_client, auth_headers, family, character_ids=[character["id"]]).json()
    mock_stream.side_effect = _stream(REPLY)

    message = _events(_send(api_client, auth_headers, conversation["id"]))[-1]["message"]

    [entry] = history.list_entries(path=str(tmp_path / "history.jsonl"))
    assert entry["id"] == message["history_id"]
    assert entry["mode"] == "conversation"
    assert entry["user_input"] == "Lena en la playa"
    assert entry["prompts"] == [{"positive_prompt": "Lena on a beach at dusk", "negative_prompt": "blurry"}]
    assert entry["cost"] == 0.003
    assert entry["params"]["conversation_id"] == conversation["id"]
    assert entry["params"]["conversation_title"] == "Lena en la playa"
    assert entry["params"]["character_names"] == ["Lena"]


@patch("app.routes.conversations.stream_chat")
def test_reply_without_prompts_is_not_logged_to_history(mock_stream, api_client, auth_headers, tmp_path):
    family = _family(tmp_path)
    conversation = _create(api_client, auth_headers, family).json()
    mock_stream.side_effect = _stream("¿Qué ambiente buscas?")

    message = _events(_send(api_client, auth_headers, conversation["id"]))[-1]["message"]

    assert message["history_id"] is None
    assert history.list_entries(path=str(tmp_path / "history.jsonl")) == []


@patch("app.routes.conversations.stream_chat")
def test_llm_gets_personality_history_and_numbered_proposals(mock_stream, api_client, auth_headers, tmp_path):
    family = _family(tmp_path)
    system_prompt.write_system_prompt(
        "conversation", "You are a grumpy art director.", path=str(tmp_path / "system_prompts.json")
    )
    conversation = _create(api_client, auth_headers, family).json()
    mock_stream.side_effect = _stream(REPLY)
    _send(api_client, auth_headers, conversation["id"])
    mock_stream.side_effect = _stream("Vale.")

    _send(api_client, auth_headers, conversation["id"], content="La 1, pero de noche")

    model, messages, temperature = mock_stream.call_args.args[1:4]
    assert model == "vendor/model"
    assert temperature == 0.5
    system = messages[0]
    assert system["role"] == "system"
    assert system["content"].startswith("You are a grumpy art director.")
    assert "family instructions" in system["content"]
    assert "Likes golden hour" in system["content"]
    # The title is already set, so the bot isn't asked for one again.
    assert "<title>" not in system["content"]
    assert [m["role"] for m in messages[1:]] == ["user", "assistant", "user"]
    assert '<prompt id="1">' in messages[2]["content"]
    assert messages[3]["content"] == "La 1, pero de noche"


@patch("app.routes.conversations.stream_chat")
def test_images_are_stored_served_and_sent_to_the_llm(mock_stream, api_client, auth_headers, tmp_path):
    family = _family(tmp_path)
    conversation = _create(api_client, auth_headers, family).json()
    mock_stream.side_effect = _stream("Se ve bien.")

    response = _send(
        api_client, auth_headers, conversation["id"], content="Mira",
        files=[("images", ("result.png", b"\x89PNG fake", "image/png"))],
    )

    assert response.status_code == 200
    stored = api_client.get(f"/api/conversations/{conversation['id']}", auth=auth_headers).json()
    [image] = stored["messages"][0]["images"]
    served = api_client.get(
        f"/api/conversations/{conversation['id']}/images/{image['id']}", auth=auth_headers
    )
    assert served.status_code == 200
    assert served.content == b"\x89PNG fake"
    assert served.headers["content-type"] == "image/png"

    user_content = mock_stream.call_args.args[2][1]["content"]
    assert user_content[0] == {"type": "text", "text": "Mira"}
    assert user_content[1]["image_url"]["url"].startswith("data:image/png;base64,")


def test_send_rejects_empty_messages_bad_images_and_too_many(api_client, auth_headers, tmp_path):
    family = _family(tmp_path)
    conversation_id = _create(api_client, auth_headers, family).json()["id"]

    assert _send(api_client, auth_headers, conversation_id, content="  ").status_code == 400
    assert _send(
        api_client, auth_headers, conversation_id,
        files=[("images", ("notes.txt", b"hello", "text/plain"))],
    ).status_code == 400
    assert _send(
        api_client, auth_headers, conversation_id,
        files=[("images", (f"{i}.png", b"x", "image/png")) for i in range(5)],
    ).status_code == 400
    # Nothing was stored by the rejected requests.
    stored = api_client.get(f"/api/conversations/{conversation_id}", auth=auth_headers).json()
    assert stored["messages"] == []


def test_send_to_unknown_conversation_is_404(api_client, auth_headers):
    response = _send(api_client, auth_headers, "00000000-0000-0000-0000-000000000000")

    assert response.status_code == 404


@patch("app.routes.conversations.stream_chat")
def test_failed_reply_keeps_the_message_and_can_be_retried(mock_stream, api_client, auth_headers, tmp_path):
    family = _family(tmp_path)
    conversation_id = _create(api_client, auth_headers, family).json()["id"]

    def failing(api_key, model, messages, temperature):
        yield "Empiezo a"
        raise RuntimeError("OpenRouter reported an error mid-reply: overloaded")

    mock_stream.side_effect = failing
    events = _events(_send(api_client, auth_headers, conversation_id))

    assert events[-1] == {"type": "error", "detail": "OpenRouter reported an error mid-reply: overloaded"}
    stored = api_client.get(f"/api/conversations/{conversation_id}", auth=auth_headers).json()
    assert [m["role"] for m in stored["messages"]] == ["user"]

    mock_stream.side_effect = _stream("Ahora sí.")
    retried = _events(api_client.post(f"/api/conversations/{conversation_id}/reply", auth=auth_headers))

    assert retried[-1]["type"] == "done"
    stored = api_client.get(f"/api/conversations/{conversation_id}", auth=auth_headers).json()
    assert [m["role"] for m in stored["messages"]] == ["user", "assistant"]


@patch("app.routes.conversations.stream_chat")
def test_empty_reply_is_an_error_and_not_stored(mock_stream, api_client, auth_headers, tmp_path):
    family = _family(tmp_path)
    conversation_id = _create(api_client, auth_headers, family).json()["id"]
    mock_stream.side_effect = _stream("  ")

    events = _events(_send(api_client, auth_headers, conversation_id))

    assert events[-1]["type"] == "error"
    stored = api_client.get(f"/api/conversations/{conversation_id}", auth=auth_headers).json()
    assert [m["role"] for m in stored["messages"]] == ["user"]


@patch("app.routes.conversations.stream_chat")
def test_retry_is_refused_when_the_last_message_has_a_reply(mock_stream, api_client, auth_headers, tmp_path):
    family = _family(tmp_path)
    conversation_id = _create(api_client, auth_headers, family).json()["id"]
    mock_stream.side_effect = _stream("Hola.")
    _send(api_client, auth_headers, conversation_id)

    response = api_client.post(f"/api/conversations/{conversation_id}/reply", auth=auth_headers)

    assert response.status_code == 409


def test_unknown_image_is_404(api_client, auth_headers, tmp_path):
    family = _family(tmp_path)
    conversation_id = _create(api_client, auth_headers, family).json()["id"]

    response = api_client.get(f"/api/conversations/{conversation_id}/images/{'0' * 32}", auth=auth_headers)

    assert response.status_code == 404

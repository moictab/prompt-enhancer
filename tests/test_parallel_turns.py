"""Turns run apart from the request that asked for them: several conversations reply at
once, a reply goes on (and is stored) with nobody following it, and any chat can pick it up."""
import json
import threading
import time
from unittest.mock import patch

from app import conversation_turn, conversations, families
from app.openrouter_client import StreamEnd


def _family(tmp_path):
    return families.create_family("SDXL", "family instructions", True, path=str(tmp_path / "families.json"))


def _conversation(api_client, auth_headers, family):
    payload = {"family_id": family["id"], "llm_model": "vendor/model", "temperature": 0.5}
    return api_client.post("/api/conversations", json=payload, auth=auth_headers).json()


def _send(api_client, auth_headers, conversation_id, content="hola"):
    return api_client.post(f"/api/conversations/{conversation_id}/messages", data={"content": content},
                           auth=auth_headers)


def _events(response):
    return [json.loads(line) for line in response.text.splitlines() if line.strip()]


def _until(condition, timeout=5.0):
    deadline = time.monotonic() + timeout
    while not condition():
        assert time.monotonic() < deadline, "timed out"
        time.sleep(0.01)


class Gated:
    """A stream_chat that says "Hola", then waits for the gate in the gated conversation."""

    def __init__(self, gated_id=None):
        self.gated_id = gated_id
        self.gate = threading.Event()

    def __call__(self, api_key, model, messages, temperature, context=None, **kwargs):
        yield "Hola"
        if self.gated_id is None or context["conversation_id"] == self.gated_id:
            assert self.gate.wait(5)
        yield " ya"
        yield StreamEnd(cost=0.001)


def _start(conversation_id, content="hola"):
    conversations.append_message(conversation_id, {"role": "user", "content": content, "images": []})
    return conversation_turn.start_turn(conversation_id)


def test_a_reply_goes_on_and_is_stored_with_nobody_following(api_client, auth_headers, tmp_path):
    conversation_id = _conversation(api_client, auth_headers, _family(tmp_path))["id"]
    stream = Gated()
    with patch("app.conversation_turn.stream_chat", stream):
        live = _start(conversation_id)
        _until(lambda: live.events)

        assert api_client.get(f"/api/conversations/{conversation_id}", auth=auth_headers).json()["replying"]
        listed = api_client.get("/api/conversations", auth=auth_headers).json()
        assert [(s["id"], s["replying"], s["commenting"]) for s in listed] == [(conversation_id, True, False)]
        # Another message would land before the reply: refused until it's done.
        assert _send(api_client, auth_headers, conversation_id).status_code == 409
        assert api_client.post(f"/api/conversations/{conversation_id}/reply", auth=auth_headers).status_code == 409

        stream.gate.set()
        _until(lambda: live.finished)

    stored = conversations.get(conversation_id)
    assert [m["role"] for m in stored["messages"]] == ["user", "assistant"]
    assert stored["messages"][-1]["content"] == "Hola ya"
    assert not conversation_turn.is_busy(conversation_id)
    assert not api_client.get(f"/api/conversations/{conversation_id}", auth=auth_headers).json()["replying"]


def test_other_conversations_reply_meanwhile(api_client, auth_headers, tmp_path):
    family = _family(tmp_path)
    slow = _conversation(api_client, auth_headers, family)["id"]
    quick = _conversation(api_client, auth_headers, family)["id"]
    stream = Gated(gated_id=slow)
    with patch("app.conversation_turn.stream_chat", stream):
        live = _start(slow)
        _until(lambda: live.events)

        events = _events(_send(api_client, auth_headers, quick))

        assert events[-1]["type"] == "done"
        assert conversation_turn.is_replying(slow)
        stream.gate.set()
        _until(lambda: live.finished)

    assert conversations.get(slow)["messages"][-1]["content"] == "Hola ya"
    assert conversations.get(quick)["messages"][-1]["content"] == "Hola ya"


def test_live_replays_the_reply_from_its_start(api_client, auth_headers, tmp_path):
    conversation_id = _conversation(api_client, auth_headers, _family(tmp_path))["id"]
    stream = Gated()
    with patch("app.conversation_turn.stream_chat", stream):
        live = _start(conversation_id)
        _until(lambda: live.events)
        threading.Timer(0.2, stream.gate.set).start()

        response = api_client.get(f"/api/conversations/{conversation_id}/live", auth=auth_headers)

    events = _events(response)
    assert [e["type"] for e in events] == ["delta", "delta", "done"]
    assert "".join(e["text"] for e in events[:2]) == "Hola ya"
    assert events[-1]["message"]["content"] == "Hola ya"


def test_live_is_204_when_nothing_is_being_written(api_client, auth_headers, tmp_path):
    conversation_id = _conversation(api_client, auth_headers, _family(tmp_path))["id"]

    assert api_client.get(f"/api/conversations/{conversation_id}/live", auth=auth_headers).status_code == 204


def test_a_quiet_turn_sends_keepalives():
    live = conversation_turn.LiveTurn("c")
    following = live.follow(keepalive=0.01)

    assert next(following) is None
    live.publish({"type": "delta", "text": "a"})
    assert next(following) == {"type": "delta", "text": "a"}
    live.finish()
    assert list(following) == []


def test_an_unexpected_failure_ends_the_turn_and_frees_the_conversation(api_client, auth_headers, tmp_path):
    conversation_id = _conversation(api_client, auth_headers, _family(tmp_path))["id"]

    def broken(*args, **kwargs):
        raise ValueError("boom")
        yield  # a generator, like stream_chat

    with patch("app.conversation_turn.stream_chat", broken):
        events = _events(_send(api_client, auth_headers, conversation_id))

    assert events == [{"type": "error", "detail": "The reply failed unexpectedly. Try again."}]
    assert not conversation_turn.is_busy(conversation_id)
    with patch("app.conversation_turn.stream_chat", Gated(gated_id="none")):
        retried = _events(api_client.post(f"/api/conversations/{conversation_id}/reply", auth=auth_headers))
    assert retried[-1]["type"] == "done"


def test_renaming_keeps_the_order_and_the_bot_stops_titling(api_client, auth_headers, tmp_path):
    conversation_id = _conversation(api_client, auth_headers, _family(tmp_path))["id"]
    before = conversations.get(conversation_id)["updated_at"]
    url = f"/api/conversations/{conversation_id}"

    renamed = api_client.patch(url, json={"title": "  Bruja   al alba "}, auth=auth_headers)

    assert renamed.status_code == 200
    assert renamed.json()["title"] == "Bruja al alba"
    assert renamed.json()["updated_at"] == before
    assert api_client.patch(url, json={"title": "   "}, auth=auth_headers).status_code == 400
    assert api_client.patch(url, json={"title": "x" * 121}, auth=auth_headers).status_code == 422

    def titled(*args, **kwargs):
        yield "<title>Otro título</title>\nVale."
        yield StreamEnd(cost=None)

    with patch("app.conversation_turn.stream_chat", titled):
        events = _events(_send(api_client, auth_headers, conversation_id))

    assert events[-1]["conversation"]["title"] == "Bruja al alba"
    assert conversations.get(conversation_id)["title"] == "Bruja al alba"


def test_summaries_say_when_the_bot_last_added_something_and_whether_runs_are_pending(
    api_client, auth_headers, tmp_path
):
    conversation_id = _conversation(api_client, auth_headers, _family(tmp_path))["id"]
    with patch("app.conversation_turn.stream_chat", Gated(gated_id="none")):
        _send(api_client, auth_headers, conversation_id)
    reply = conversations.get(conversation_id)["messages"][-1]

    listed = api_client.get("/api/conversations", auth=auth_headers).json()[0]

    assert listed["last_reply_at"] == reply["timestamp"]
    assert listed["running"] is False
    conversations.add_run(conversation_id, {"workflow": "w", "status": "pending", "jobs": []}, [])
    assert api_client.get("/api/conversations", auth=auth_headers).json()[0]["running"] is True


def test_summaries_carry_the_newest_run_images_with_when_they_arrived(api_client, auth_headers, tmp_path,
                                                                       monkeypatch):
    """For the list to show the images that came since the user last opened the conversation."""
    from .conftest import png_bytes
    monkeypatch.setattr(conversations, "RECENT_IMAGES", 3)
    conversation_id = _conversation(api_client, auth_headers, _family(tmp_path))["id"]
    shared = conversations.save_image(conversation_id, png_bytes(), "image/png")
    conversations.append_message(conversation_id, {"role": "user", "content": "mira", "images": [shared]})
    made = [conversations.save_image(conversation_id, png_bytes(size=(2400, 2400)), "image/png") for _ in range(4)]
    conversations.append_message(conversation_id, {"role": "results", "run": 1, "content": "", "images": made[:2]})
    conversations.append_message(conversation_id, {"role": "results", "run": 2, "content": "", "images": made[2:]})
    first, second = conversations.get(conversation_id)["messages"][-2:]

    listed = api_client.get("/api/conversations", auth=auth_headers).json()[0]["recent_images"]

    assert [i["id"] for i in listed] == [made[3]["id"], made[2]["id"], made[1]["id"]]  # newest first, runs' only
    assert [i["at"] for i in listed] == [second["timestamp"], second["timestamp"], first["timestamp"]]
    assert all(i["small"] for i in listed)

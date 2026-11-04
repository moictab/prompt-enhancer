"""Notes by scope: general everywhere, a family's where that family is used, a topic's where the
conversation is about it (declared by the bot, or named in the latest messages) -- within a budget."""
from unittest.mock import patch

from app import chat, conversations, families, memory

from .conftest import message_text
from .test_api_conversar_comfy import Script, _conversation, _send


def test_the_bot_writes_scoped_notes_confirms_them_and_declares_topics():
    reply = chat.parse_reply(
        '<remember family="Krea 2">negations don\'t suppress (2 of 2)</remember>'
        "<remember topic='Kim'>freckled heart-shaped face</remember><remember>likes dusk light</remember>"
        '<remember id="4" topic="Kim">thick smudged eyeliner</remember><confirm id="2"/>'
        "<topics>Kim, Ramona</topics>Hecho.", True)
    assert reply.notes == ["negations don't suppress (2 of 2)", "freckled heart-shaped face", "likes dusk light"]
    assert [memory.clean_scope(**m) for m in reply.note_meta] == [
        {"scope": "family", "family": "krea-2"}, {"scope": "topic", "topic": "Kim"}, {"scope": "general"}]
    assert reply.note_updates == [(4, "thick smudged eyeliner")] and reply.update_meta[4]["topic"] == "Kim"
    assert reply.confirms == [2] and reply.topics == ["Kim", "Ramona"] and reply.content == "Hecho."


def test_a_turn_carries_what_applies_within_the_budget():
    general = memory.create_note("likes dusk light")
    krea = memory.create_note("negations don't suppress", scope={"scope": "family", "family": "krea-2"})
    memory.create_note("legs pass through chairs", scope={"scope": "family", "family": "anima"})
    kim = memory.create_note("freckled heart-shaped face", scope={"scope": "topic", "topic": "Kim"})
    memory.create_note("short bob, red", scope={"scope": "topic", "topic": "Ramona"})
    memory.set_topic_keywords("Kim", ["Kim Pine"])

    covers = lambda folder: folder == "krea-2"  # noqa: E731
    # Named in the messages (a whole word, accents and case aside), or declared: active.
    assert memory.active_topics([], ["una foto de KIM PINE"]) == ["Kim"]
    assert memory.active_topics([], ["un kimono rojo"]) == []
    assert memory.active_topics(["ramona"], []) == ["Ramona"]
    shown, left = memory.notes_for_turn(memory.list_notes(), covers, ["Kim"])
    assert [n["id"] for n in shown] == [general["id"], krea["id"], kim["id"]]
    assert left == {"family anima": 1, "topic Ramona": 1}
    text = chat.notes_text(shown, {"left": left, "topics": ["Kim"], "known": memory.topic_names()})
    assert text.splitlines()[:6] == ["General:", "[1] likes dusk light", "Family krea-2 (this conversation's family):",
                                     "[2] negations don't suppress", "Topic Kim:", "[3] freckled heart-shaped face"]
    assert "family anima (1), topic Ramona (1)" in text and "Topics your notes have: Kim, Ramona." in text
    # Over the budget: the rest is named, not sent.
    shown, left = memory.notes_for_turn(memory.list_notes(), covers, ["Kim"], budget=60)
    assert len(shown) == 2 and left["topic Kim"] == 1


def test_in_a_conversation_the_bot_sees_its_scopes_and_its_choices_stick(api_client, auth_headers, tmp_path):
    family = families.create_family("Krea 2", "rules", False, path=str(tmp_path / "families.json"))
    conversation_id = _conversation(api_client, auth_headers, family)["id"]
    memory.create_note("likes dusk light")
    krea = memory.create_note("negations don't suppress", scope={"scope": "family", "family": "krea-2"})
    memory.create_note("legs pass through chairs", scope={"scope": "family", "family": "anima"})
    memory.create_note("freckled heart-shaped face", scope={"scope": "topic", "topic": "Kim"})
    script = Script(('<topics>kim</topics><confirm id="2"/><remember topic="Kim">thick eyeliner</remember>Va.', []),
                    ("Sigo.", []))
    with patch("app.conversation_turn.stream_chat", script):
        _send(api_client, auth_headers, conversation_id, "una foto de Kim en su cuarto")
        _send(api_client, auth_headers, conversation_id, "otra, más luz")  # Kim isn't named: declared now

    first, second = (message_text(call["messages"][-1]) for call in script.calls)
    for sent in (first, second):
        assert "[2] negations don't suppress" in sent and "freckled heart-shaped face" in sent
        assert "legs pass through chairs" not in sent and "family anima (1)" in sent
    assert "This conversation has no topics yet" in first and "This conversation's topics: Kim." in second
    stored = conversations.get(conversation_id)
    assert stored["topics"] == ["Kim"]
    assert next(n for n in memory.list_notes() if n["id"] == krea["id"])["confirmations"] == 1
    added = next(n for n in memory.list_notes() if n["text"] == "thick eyeliner")
    assert (added["scope"], added["topic"]) == ("topic", "Kim")
    assert stored["messages"][1]["notes"][0]["scope"] == "topic Kim"


def test_ajustes_moves_notes_between_scopes_and_keeps_topic_keywords(api_client, auth_headers):
    created = api_client.post("/api/admin/memory", auth=auth_headers,
                              json={"text": "short bob", "scope": "topic", "topic": "Ramona"}).json()
    assert (created["scope"], created["topic"]) == ("topic", "Ramona")
    moved = api_client.put(f"/api/admin/memory/{created['id']}", auth=auth_headers,
                           json={"text": "short bob", "scope": "family", "family": "Krea 2"}).json()
    assert (moved["scope"], moved["family"]) == ("family", "krea-2") and "topic" not in moved
    kept = api_client.put(f"/api/admin/memory/{created['id']}", auth=auth_headers, json={"text": "bob"}).json()
    assert kept["family"] == "krea-2"  # a rewrite without a scope keeps it
    assert api_client.post("/api/admin/memory", auth=auth_headers,
                           json={"text": "x", "scope": "topic"}).status_code == 400
    api_client.post("/api/admin/memory", auth=auth_headers, json={"text": "pecas", "topic": "Kim"})
    assert api_client.put("/api/admin/memory/topics/Kim", auth=auth_headers,
                          json={"keywords": ["Kim Pine", " ", "Kim Pine"]}).json()["keywords"] == ["Kim Pine"]
    assert api_client.get("/api/admin/memory/topics", auth=auth_headers).json() == [
        {"name": "Kim", "keywords": ["Kim Pine"]}]
    assert api_client.put("/api/admin/memory/topics/Nadie", auth=auth_headers,
                          json={"keywords": []}).status_code == 404

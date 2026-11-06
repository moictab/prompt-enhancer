"""The user's words about run images ("they're all great") recorded as their reactions, by the bot."""
from unittest.mock import patch

from app import balance, catalog, chat, conversations, runs, workflows

from .conftest import message_text
from .test_api_conversar_comfy import Script, _call, _conversation, _family, _send

LAUNCH = _call("run_workflow", workflow="krea-2/txt2img", changes={"positive": "a witch at dawn"}, count=3)


def _deliver(fake_comfy, *prompt_ids):
    for prompt_id in prompt_ids:
        fake_comfy.finish(prompt_id, f"{prompt_id}.png")
    runs.tracker.poll_once()
    runs.wait_until_idle()


def test_words_about_run_images_count_like_thumbs(api_client, auth_headers, tmp_path, fake_comfy):
    workflows.ensure_seeded()
    conversation_id = _conversation(api_client, auth_headers, _family(tmp_path), auto_comment=False)["id"]
    script = Script(("Van tres brujas:", [LAUNCH]),
                    ('¡Me alegro! <rating images="1, 2,3" value="up">Espectaculares, son todas geniales</rating>'
                     "Sigo por ahí.", []))
    with patch("app.conversation_turn.stream_chat", script):
        _send(api_client, auth_headers, conversation_id, "hazme una bruja")
        _deliver(fake_comfy, "prompt-1", "prompt-2", "prompt-3")
        images = conversations.get(conversation_id)["messages"][-1]["images"]
        # The user disliked the second one with the thumb: that stands.
        api_client.put(f"/api/conversations/{conversation_id}/images/{images[1]['id']}/reaction", auth=auth_headers,
                       json={"rating": "down", "comment": "manos raras"})
        _send(api_client, auth_headers, conversation_id, "Espectaculares, son todas geniales")

    stored = conversations.get(conversation_id)
    reactions = [img.get("reaction") for img in next(m for m in stored["messages"] if m["role"] == "results")["images"]]
    assert reactions[0]["rating"] == "up" and reactions[0]["inferred"] and reactions[0]["sent"]
    assert reactions[0]["comment"] == "Espectaculares, son todas geniales"
    assert reactions[1]["rating"] == "down" and not reactions[1].get("inferred")
    assert reactions[2]["rating"] == "up" and reactions[2]["inferred"]
    reply = stored["messages"][-1]
    assert [r["number"] for r in reply["ratings"]] == [1, 3] and "rating" not in reply["content"]
    # They count like thumbs: in the models' stats and in the balance.
    stats = catalog.stats()
    model = next(record for record in stats.values() if record.get("images"))
    assert model["liked"] == 2 and model["disliked"] == 1
    ledger = balance.ledger(stored)
    assert ledger.counts["liked"] == 2 and "user liked (said so in a message)" in ledger.text
    # The bot isn't sent back what it recorded itself.
    assert conversations.unsent_reactions(stored) == []
    assert '<rating images="5,6,7,8" value="up">' in chat.COMFY_MECHANICS


def test_only_run_images_take_a_rating_and_a_thumb_replaces_it(api_client, auth_headers, tmp_path, fake_comfy):
    workflows.ensure_seeded()
    conversation_id = _conversation(api_client, auth_headers, _family(tmp_path), auto_comment=False)["id"]
    script = Script(("Va:", [LAUNCH]), ('<rating images="1,9" value="down">no me gusta el sombrero</rating>Lo quito.', []))
    with patch("app.conversation_turn.stream_chat", script):
        _send(api_client, auth_headers, conversation_id, "hazme una bruja")
        _deliver(fake_comfy, "prompt-1", "prompt-2", "prompt-3")
        _send(api_client, auth_headers, conversation_id, "no me gusta el sombrero de la primera")
    stored = conversations.get(conversation_id)
    first = next(m for m in stored["messages"] if m["role"] == "results")["images"][0]
    assert first["reaction"]["rating"] == "down" and first["reaction"]["inferred"]
    assert [r["number"] for r in stored["messages"][-1]["ratings"]] == [1]  # no image 9
    # The user's thumb replaces the bot's reading of their words.
    changed = api_client.put(f"/api/conversations/{conversation_id}/images/{first['id']}/reaction",
                             auth=auth_headers, json={"rating": "up", "comment": ""}).json()
    assert changed["reaction"]["rating"] == "up" and not changed["reaction"].get("inferred")


def test_a_comment_left_without_a_rating_is_their_words_too(api_client, auth_headers, tmp_path, fake_comfy):
    workflows.ensure_seeded()
    conversation_id = _conversation(api_client, auth_headers, _family(tmp_path), auto_comment=False)["id"]
    script = Script(("Van tres:", [LAUNCH]), ('<rating images="1" value="up">espectacular</rating>¡Bien!', []))
    put = f"/api/conversations/{conversation_id}/images"
    with patch("app.conversation_turn.stream_chat", script):
        _send(api_client, auth_headers, conversation_id, "hazme una bruja")
        _deliver(fake_comfy, "prompt-1", "prompt-2", "prompt-3")
        images = next(m for m in conversations.get(conversation_id)["messages"] if m["role"] == "results")["images"]
        api_client.put(f"{put}/{images[0]['id']}/reaction", auth=auth_headers, json={"rating": None, "comment": "genial"})
        _send(api_client, auth_headers, conversation_id, "")

    assert 'image 1, a comment without a rating: "genial"' in message_text(script.calls[1]["messages"][-1])
    first = next(m for m in conversations.get(conversation_id)["messages"] if m["role"] == "results")["images"][0]
    # Rated by it, keeping their own words.
    assert first["reaction"]["rating"] == "up" and first["reaction"]["inferred"]
    assert first["reaction"]["comment"] == "genial"
    assert next(record for record in catalog.stats().values() if record.get("images"))["liked"] == 1
    # Deshacer: the comment stays, as the bot already had it -- not sent again.
    undone = api_client.put(f"{put}/{images[0]['id']}/reaction", auth=auth_headers,
                            json={"rating": None, "comment": "genial"}).json()["reaction"]
    assert undone["rating"] is None and undone["comment"] == "genial" and undone["sent"]
    assert conversations.unsent_reactions(conversations.get(conversation_id)) == []


def test_a_like_says_what_they_liked_and_the_bot_notes_what_is_wrong_all_the_same(
        api_client, auth_headers, tmp_path, fake_comfy):
    workflows.ensure_seeded()
    conversation_id = _conversation(api_client, auth_headers, _family(tmp_path), auto_comment=False)["id"]
    script = Script(("Van tres:", [LAUNCH]),
                    ('Apunto las manos. <flaws images="1,2,3">hands, vibes</flaws>Sigo.', []))
    put = f"/api/conversations/{conversation_id}/images"
    with patch("app.conversation_turn.stream_chat", script):
        _send(api_client, auth_headers, conversation_id, "hazme una bruja")
        _deliver(fake_comfy, "prompt-1", "prompt-2", "prompt-3")
        images = next(m for m in conversations.get(conversation_id)["messages"] if m["role"] == "results")["images"]
        liked = api_client.put(f"{put}/{images[0]['id']}/reaction", auth=auth_headers, json={
            "rating": "up", "comment": "está bien, pero las manos han salido mal",
            "reasons": ["face", "light", "hands"], "flaws": ["face"]}).json()["reaction"]
        # A like's reasons are what they liked (a dislike's are dropped); flaws only the bot sets.
        assert liked["reasons"] == ["face", "light"] and "flaws" not in liked
        # A dislike needs a reason of its own: what they'd like isn't one.
        assert api_client.put(f"{put}/{images[1]['id']}/reaction", auth=auth_headers,
                              json={"rating": "down", "reasons": ["light"]}).status_code == 400
        api_client.put(f"{put}/{images[1]['id']}/reaction", auth=auth_headers,
                       json={"rating": "down", "reasons": ["hands"]})
        _send(api_client, auth_headers, conversation_id, "")

    assert 'image 1 liked (for: face, light): "está bien, pero las manos han salido mal"' in \
        message_text(script.calls[1]["messages"][-1])
    stored = conversations.get(conversation_id)
    first, second, third = next(m for m in stored["messages"] if m["role"] == "results")["images"]
    # Only on the image they liked: not the disliked one, not the unrated one; unknown words dropped.
    assert first["reaction"]["flaws"] == ["hands"] and first["reaction"]["rating"] == "up"
    assert first["reaction"]["sent"] and "flaws" not in second["reaction"] and "reaction" not in third
    assert stored["messages"][-1]["flaws"] == [{"image_id": first["id"], "number": 1, "flaws": ["hands"]}]
    assert "<flaws" not in stored["messages"][-1]["content"]
    model = next(record for record in catalog.stats().values() if record.get("images"))
    assert model["liked_for"] == {"face": 1, "light": 1} and model["liked_flaws"] == {"hands": 1}
    assert "1 liked (for face 1, light 1; but hands 1)" in catalog._record_text(model)
    assert "user liked (for face, light; but hands)" in balance.ledger(stored).text

    # The user takes a flaw away: nothing new for the bot.
    kept = api_client.put(f"{put}/{first['id']}/reaction", auth=auth_headers, json={
        "rating": "up", "comment": "está bien, pero las manos han salido mal", "reasons": ["face", "light"],
        "flaws": []}).json()["reaction"]
    assert "flaws" not in kept and kept["sent"]
    assert conversations.unsent_reactions(conversations.get(conversation_id)) == []


def test_a_favorite_is_the_strongest_like_and_a_dislike_can_be_a_tap(api_client, auth_headers, tmp_path, fake_comfy):
    workflows.ensure_seeded()
    conversation_id = _conversation(api_client, auth_headers, _family(tmp_path), auto_comment=False)["id"]
    script = Script(("Van tres:", [LAUNCH]), ("Me centro en la 1.", []),
                    ('<rating images="3" value="love">¡esta es perfecta!</rating>Anotado.', []))
    put = f"/api/conversations/{conversation_id}/images"
    with patch("app.conversation_turn.stream_chat", script):
        _send(api_client, auth_headers, conversation_id, "hazme una bruja")
        _deliver(fake_comfy, "prompt-1", "prompt-2", "prompt-3")
        images = next(m for m in conversations.get(conversation_id)["messages"] if m["role"] == "results")["images"]
        assert api_client.put(f"{put}/{images[0]['id']}/reaction", auth=auth_headers,
                              json={"rating": "love"}).status_code == 200
        # A dislike with a tap on what failed, no words needed; without either, refused.
        assert api_client.put(f"{put}/{images[1]['id']}/reaction", auth=auth_headers,
                              json={"rating": "down"}).status_code == 400
        tapped = api_client.put(f"{put}/{images[1]['id']}/reaction", auth=auth_headers,
                                json={"rating": "down", "reasons": ["hands", "face", "hands"]})
        assert tapped.json()["reaction"]["reasons"] == ["hands", "face"]
        assert api_client.put(f"{put}/{images[1]['id']}/reaction", auth=auth_headers,
                              json={"rating": "down", "reasons": ["vibes"]}).status_code == 422
        _send(api_client, auth_headers, conversation_id, "")  # just the reactions
        _send(api_client, auth_headers, conversation_id, "¡la tercera es perfecta!")

    sent = message_text(script.calls[1]["messages"][-1])
    assert "image 1 loved (a favorite)" in sent and "image 2 disliked (hands, face)" in sent
    stored = conversations.get(conversation_id)
    third = next(m for m in stored["messages"] if m["role"] == "results")["images"][2]
    assert third["reaction"]["rating"] == "love" and third["reaction"]["inferred"]
    model = next(record for record in catalog.stats().values() if record.get("images"))
    assert (model["liked"], model["loved"], model["disliked"]) == (2, 2, 1)
    assert model["dislike_reasons"] == {"hands": 1, "face": 1}
    assert "2 liked (2 a favorite)" in catalog._record_text(model) and "1 disliked (hands 1, face 1)" in \
        catalog._record_text(model)
    ledger = balance.ledger(stored)
    assert (ledger.counts["liked"], ledger.counts["loved"], ledger.counts["unrated"]) == (2, 2, 0)
    assert "user disliked (hands, face)" in ledger.text and "user loved (a favorite)" in ledger.text

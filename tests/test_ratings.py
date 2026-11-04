"""The user's words about run images ("they're all great") recorded as their reactions, by the bot."""
from unittest.mock import patch

from app import balance, catalog, chat, conversations, runs, workflows

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

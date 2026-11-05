"""The away mode ("Modo autónomo"): while the user is away the bot works on its own in a conversation of its
own, at idle GPU priority and within what Claude Code's 5-hour window can spare without leaving the user short
when they're back -- and it ends as soon as they do anything that makes the bot work."""
import time
from unittest.mock import patch

import pytest

from app import away, claude_code, conversation_turn, conversations, retry_later, runs, storage, workflows

from .conftest import png_bytes
from .test_api_conversar_comfy import Script, _conversation, _family, _send
from .test_claude_code import MODEL, RATE_LIMIT, RUN_INPUT, claude, init, result, text, tool  # noqa: F401

HOUR = 3600
LORA = "krea-2\\style-ink_v1.safetensors"


@pytest.fixture(autouse=True)
def no_timers_left():
    yield
    away._cancel_end()
    for conversation_id in retry_later.pending():
        retry_later.cancel(conversation_id)


def _limits(**windows):
    storage.write_json(str(claude_code.limits_path()), {"status": "allowed", "windows": windows})


def _favorite(family, rating="love", comment="justo así"):
    """A conversation whose run made an image the user marked: the away mode starts from it."""
    workflows.ensure_seeded()
    graph = workflows.find_by_name("krea-2/txt2img")["workflow"]
    source = conversations.create(family["id"], "vendor/model", 0.7)
    image = conversations.save_image(source["id"], png_bytes(), "image/png", workflow=graph)
    conversations.add_run(source["id"], {
        "workflow": "krea-2/txt2img", "changes": {}, "count": 1, "status": "done",
        "jobs": [{"prompt_id": "p-old", "status": "done", "images": [image["id"]],
                  "loras": [{"name": LORA, "strength": 0.6}]}]}, [graph])
    conversations.append_message(source["id"], {"role": "results", "run": 1, "content": "", "images": [image]})
    conversations.update_settings(source["id"], {"title": "Bruja en el bosque"})
    conversations.set_reaction(source["id"], image["id"], rating, comment)
    return source


def _start(api_client, auth_headers, hours=8, **fields):
    return api_client.post("/api/conversations/away", auth=auth_headers,
                           json={"until": time.time() + hours * HOUR, **fields})


def _settle(conversation_id):
    live = conversation_turn.live_turn(conversation_id)
    if live is not None:
        for _ in live.follow(keepalive=0.05):
            pass


# --- the rule: 5-hour windows ----------------------------------------------------------------------------

def test_turns_use_a_window_that_renews_before_the_user_is_back_but_a_quarter_of_the_one_still_open():
    now = 1_000_000.0
    working = {"until": now + 9 * HOUR}
    # No window open: a turn opens one, which renews 5 h later, before the user is back: nearly all of it.
    seen = away.outlook(working, None, now)
    assert (seen.action, seen.turns, seen.capped) == ("go", 19, False)
    # An open one renewing in an hour, half used: what's left, with more after it renews.
    half = {"five_hour": {"used": 0.5, "resets_at": now + HOUR}}
    seen = away.outlook(working, {"windows": half}, now)
    assert (seen.action, seen.turns, seen.renews) == ("go", 9, True)
    # Out of room in it: wait for it to renew.
    full = {"five_hour": {"used": 0.93, "resets_at": now + HOUR}}
    assert (away.outlook(working, {"windows": full}, now).action, away.outlook(working, {"windows": full}, now).at) \
        == ("wait", now + HOUR)

    # Back in 3 h: a window opened now is still open then, so it gets a quarter of it at most.
    soon = {"until": now + 3 * HOUR}
    seen = away.outlook(soon, None, now)
    assert (seen.action, seen.turns, seen.capped) == ("go", 5, True)
    # Room for this one turn only: the last, which sums up.
    used = {"five_hour": {"used": 0.16, "resets_at": now + 4 * HOUR}}
    assert (away.outlook(soon, {"windows": used}, now).action, away.outlook(soon, {"windows": used}, now).reason) \
        == ("last", "limit")
    # Not even that: it ends, leaving the user 75% at least.
    spent = {"five_hour": {"used": 0.22, "resets_at": now + 4 * HOUR}}
    assert (away.outlook(soon, {"windows": spent}, now).action, away.outlook(soon, {"windows": spent}, now).reason) \
        == ("end", "limit")


def test_the_time_ends_it_with_a_last_turn_and_the_week_is_never_run_out():
    now = 1_000_000.0
    assert away.outlook({"until": now + 10 * 60}, None, now).action == "last"
    assert away.outlook({"until": now - 1}, None, now).reason == "until"
    working = {"until": now + 9 * HOUR}
    week = {"seven_day": {"used": 0.93, "resets_at": now + 2 * HOUR}}
    assert away.outlook(working, {"windows": week}, now).action == "wait"  # it renews tonight
    week["seven_day"]["resets_at"] = now + 30 * HOUR
    assert away.outlook(working, {"windows": week}, now).reason == "limit"
    week["seven_day"]["used"] = 0.6  # no ceiling of its own
    assert away.outlook(working, {"windows": week}, now).action == "go"


def test_a_turns_share_is_the_largest_recent_rise_within_a_window():
    def reply(used, resets):
        return {"role": "assistant", "claude_code": {"rate_limit": {"unifiedWindows": {
            "five_hour": {"utilization": used, "resetsAt": resets}}}}}

    conversation = {"messages": [reply(0.10, 1), reply(0.12, 1), reply(0.20, 1), {"role": "results"},
                                 reply(0.02, 2), reply(0.05, 2)]}
    assert away.turn_share(conversation) == pytest.approx(0.08)
    assert away.turn_share({"messages": []}) == away.TURN_SHARE


# --- the mode -----------------------------------------------------------------------------------------------

def test_it_starts_a_conversation_of_its_own_from_the_favorites_and_launches_at_idle(
        api_client, auth_headers, tmp_path, fake_comfy, claude):
    family = _family(tmp_path)
    _favorite(family)
    fake_comfy.model_files["diffusion_models"] = [MODEL]
    fake = claude([init(), text("Esta noche, variaciones de #1."), tool("c1", "run_workflow", RUN_INPUT),
                   RATE_LIMIT, result()])

    response = _start(api_client, auth_headers, instructions="céntrate en brujas", claude_model="sonnet")
    assert response.status_code == 200, response.text
    started = response.json()
    _settle(started["id"])

    stored = conversations.get(started["id"])
    assert stored["title"].startswith("Modo autónomo") and stored["engine"] == "claude-code"
    assert stored["claude_model"] == "sonnet" and stored["auto_comment"] is False
    brief = stored["messages"][0]
    assert brief["away"]["favorites"] == 1 and len(brief["images"]) == 1
    # The favorite comes in with its workflow: the bot can run it (image:1) or finish it.
    assert conversations.image_workflow(started["id"], brief["images"][0]["id"]) is not None
    sent = fake.sent()
    assert "[Away mode]" in sent and "What the user asked for this time: céntrate en brujas" in sent
    assert 'a favorite (loved), from the conversation "Bruja en el bosque"' in sent and "justo así" in sent
    assert "style-ink_v1" in sent  # the LoRA it uses, never calibrated
    assert "## Away Mode" in sent and "about 19 turns left" in sent and "Start:" in sent
    # By default the night is half and half: their favorites, and surprising them from a few sparks.
    assert "Split the night" in sent and "Sparks for tonight:" in sent
    assert brief["away"]["focus"] == "mix" and len(brief["away"]["sparks"]) == away.SPARKS["mix"]
    # Nothing it launches goes ahead of anyone.
    assert fake_comfy.priorities == ["idle"] and "modo autónomo" in fake_comfy.tags[0]
    assert stored["autonomy"]["kind"] == "away" and stored["autonomy"]["active"]
    assert stored["autonomy"]["images"] == 1
    listed = next(s for s in api_client.get("/api/conversations", auth=auth_headers).json() if s["id"] == started["id"])
    assert listed["autonomy"]["kind"] == "away" and listed["autonomy"]["until"] == stored["autonomy"]["until"]
    # Only one at a time.
    assert _start(api_client, auth_headers).status_code == 409

    # The user writes somewhere else: they're back, and it ends -- its run goes on, at idle.
    other = _conversation(api_client, auth_headers, family)
    with patch("app.conversation_turn.stream_chat", Script(("Hola.", []))):
        _send(api_client, auth_headers, other["id"], "hola")
    ended = conversations.get(started["id"])["autonomy"]
    assert not ended["active"] and ended["end_reason"] == "user"
    assert away.current() is None and fake_comfy.cancelled == []
    # Its images land without a comment: no other Claude Code turn.
    fake_comfy.finish("prompt-1", "away.png")
    runs.tracker.poll_once()
    runs.wait_until_idle()
    assert conversations.get(started["id"])["messages"][-1]["role"] == "results"
    assert len(fake.launches) == 1


def test_surprising_them_starts_from_random_sparks_not_their_favorites(
        api_client, auth_headers, tmp_path, fake_comfy, claude):
    family = _family(tmp_path)
    _favorite(family)
    other = _conversation(api_client, auth_headers, family)
    conversations.update_settings(other["id"], {"title": "Ramona en la playa"})
    conversations.append_message(other["id"], {"role": "user", "content": "hola", "images": []})
    storage.write_json(str(tmp_path / "catalog.json"), {"entries": {
        LORA: {"folder": "loras", "family": "krea-2"},
        "krea-2\\concept-paper_v1.safetensors": {"folder": "loras", "family": "krea-2"}}})
    fake = claude([init(), text("Primera idea: un faro de papel."), result()],
                  [init(), text("Otra noche."), result()])

    started = _start(api_client, auth_headers, focus="explore").json()
    _settle(started["id"])

    brief = conversations.get(started["id"])["messages"][0]
    # Nothing to anchor it to their favorites: no images copied, four cards per spark, five sparks.
    assert brief["images"] == [] and brief["away"]["focus"] == "explore"
    assert len(brief["away"]["sparks"]) == away.SPARKS["explore"]
    assert all(len(spark) == 4 for spark in brief["away"]["sparks"])
    sent = fake.sent()
    assert "Tonight is for surprising them" in sent and "Sparks for tonight:" in sent
    assert "; ".join(brief["away"]["sparks"][0]) in sent
    assert '"Ramona en la playa"' in sent  # what they usually make: to step away from
    assert "concept-paper_v1" in sent  # installed and never used here
    assert "Their favorites --" not in sent and "Tonight is for surprising them: new ideas" in sent
    stored = conversations.get(started["id"])
    assert stored["autonomy"]["focus"] == "explore"

    # The next night draws other cards.
    away.stop(started["id"], "until")
    second = _start(api_client, auth_headers, focus="explore").json()
    _settle(second["id"])
    tonight = {card for spark in conversations.get(second["id"])["messages"][0]["away"]["sparks"] for card in spark}
    last_night = {card for spark in brief["away"]["sparks"] for card in spark}
    assert not tonight & last_night


def test_sparks_differ_within_a_draw_and_skip_recent_cards():
    import random

    from app import sparks
    drawn = sparks.draw(5, rng=random.Random(7))
    cards = [card for spark in drawn for card in spark]
    assert len(cards) == len(set(cards)) == 20
    again = sparks.draw(5, avoid=cards, rng=random.Random(7))
    assert not {card for spark in again for card in spark} & set(cards)
    # With a deck used up, it draws from all of it again rather than nothing.
    assert len(sparks.draw(1, avoid=sparks.SUBJECTS)[0]) == 4


def test_out_of_room_in_the_window_it_waits_for_it_to_renew(api_client, auth_headers, tmp_path, fake_comfy, claude):
    _family(tmp_path)
    resets = int(time.time()) + HOUR
    _limits(five_hour={"used": 0.94, "resets_at": resets, "status": "allowed_warning"})
    fake = claude()

    started = _start(api_client, auth_headers).json()
    _settle(started["id"])

    stored = conversations.get(started["id"])
    assert stored["retry_at"] == resets and started["id"] in retry_later.pending()
    assert fake.launches == [] and stored["autonomy"]["active"]
    # "Detener": it ends, and won't come back when the window renews.
    api_client.post(f"/api/conversations/{started['id']}/autonomy/stop", auth=auth_headers)
    stored = conversations.get(started["id"])
    assert "retry_at" not in stored and started["id"] not in retry_later.pending()
    assert stored["autonomy"]["end_reason"] == "user"


def test_it_wont_start_with_no_room_before_the_user_is_back(api_client, auth_headers, tmp_path, fake_comfy, claude):
    _family(tmp_path)
    claude()
    _limits(five_hour={"used": 0.99, "resets_at": int(time.time()) + 6 * HOUR})
    response = _start(api_client, auth_headers, hours=3)
    assert response.status_code == 409 and "99%" in response.json()["detail"]
    assert _start(api_client, auth_headers, hours=0.2).status_code == 409  # too soon to be worth it


def test_its_last_turn_sums_up_and_ends_it(api_client, auth_headers, tmp_path, fake_comfy, claude, monkeypatch):
    _family(tmp_path)
    monkeypatch.setattr(away, "FINAL_SECONDS", 10 * HOUR)  # already within the last stretch
    fake = claude([init(), text("Resumen de la noche: lo mejor, #1."), result()])

    started = _start(api_client, auth_headers).json()
    _settle(started["id"])

    assert "This is your last turn" in fake.sent()
    stored = conversations.get(started["id"])
    assert stored["messages"][-1]["role"] == "assistant"
    assert not stored["autonomy"]["active"] and stored["autonomy"]["end_reason"] == "until"
    assert away.current() is None

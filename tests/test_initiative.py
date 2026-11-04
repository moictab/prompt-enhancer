"""The bot's initiative: after each of the user's messages it may carry on alone -- launch, look
at its images, launch the next batch -- within the conversation's image budget."""
from unittest.mock import patch

from app import chat, conversations, runs, workflows

from .conftest import message_text
from .test_api_conversar_comfy import Script, _call, _conversation, _family, _send


def _launch(count, call_id="call-1", prompt="a witch at dawn"):
    return _call("run_workflow", call_id, workflow="krea-2/txt2img", changes={"positive": prompt}, count=count)


def _deliver(fake_comfy, *prompt_ids):
    for prompt_id in prompt_ids:
        fake_comfy.finish(prompt_id, f"{prompt_id}.png")
    runs.tracker.poll_once()
    runs.wait_until_idle()


def test_after_a_message_the_bot_goes_on_alone_until_the_budget_is_spent(api_client, auth_headers, tmp_path,
                                                                        fake_comfy):
    workflows.ensure_seeded()
    conversation_id = _conversation(api_client, auth_headers, _family(tmp_path), initiative_images=3)["id"]
    script = Script(("Exploro dos direcciones:", [_launch(2)]),       # the user's turn
                    ("La 2 promete; la empujo:", [_launch(1, "call-2", "a witch at dawn, closer")]),  # its own look
                    ("Resumen: la mejor es #3.", []))                   # out of budget: it can only sum up

    with patch("app.conversation_turn.stream_chat", script):
        _send(api_client, auth_headers, conversation_id, "hazme una bruja")
        working = conversations.autonomy(conversations.get(conversation_id))
        assert (working["kind"], working["images"], working["max_images"]) == ("initiative", 2, 3)
        _deliver(fake_comfy, "prompt-1", "prompt-2")
        _deliver(fake_comfy, "prompt-3")

    user_turn, own_look, last_look = script.calls
    assert "Between the user's messages you carry on alone" in message_text(user_turn["messages"][-1])
    # Its look at the first batch could launch, with no one asking.
    assert own_look["tools"] and own_look["tool_choice"] is None
    assert "launch it in this same reply, without asking" in message_text(own_look["messages"][-1])
    # The budget spent, the last look launches nothing.
    assert last_look["tool_choice"] == "none"
    assert "The budget is spent" in message_text(last_look["messages"][-1])
    # What it launched by itself waits behind what someone asked for.
    assert fake_comfy.priorities == ["high", "high", "low"]
    # comfy-gateway's panel says which conversation, which run and who it is for.
    assert fake_comfy.tags == ["hazme una bruja · ejecución 1 · tras tu mensaje"] * 2 + [
        "hazme una bruja · ejecución 2 · por su cuenta"]
    ended = conversations.get(conversation_id)["autonomy"]
    assert ended["active"] is False and ended["end_reason"] == "images" and ended["images"] == 3

    # The user's next message brings a new budget.
    with patch("app.conversation_turn.stream_chat", Script(("Vale, ¿qué más?", []))):
        _send(api_client, auth_headers, conversation_id, "me gusta la 3")
    again = conversations.get(conversation_id)["autonomy"]
    assert (again["kind"], again["images"], again["end_reason"]) == ("initiative", 0, "done")


def test_with_no_budget_the_bot_launches_only_what_it_is_asked(api_client, auth_headers, tmp_path, fake_comfy):
    workflows.ensure_seeded()
    conversation_id = _conversation(api_client, auth_headers, _family(tmp_path), initiative_images=0)["id"]
    script = Script(("Va:", [_launch(1)]), ("La luz funciona; probaría niebla.", []))

    with patch("app.conversation_turn.stream_chat", script):
        _send(api_client, auth_headers, conversation_id, "hazme una bruja")
        _deliver(fake_comfy, "prompt-1")

    look = script.calls[1]
    assert look["tool_choice"] == "none" and chat.LOOKING_RULE in message_text(look["messages"][-1])
    assert "autonomy" not in conversations.get(conversation_id)


def test_a_task_it_was_given_survives_the_users_messages_and_ending_stops_the_budget(api_client, auth_headers,
                                                                                    tmp_path, fake_comfy):
    workflows.ensure_seeded()
    family = _family(tmp_path)
    tasked = _conversation(api_client, auth_headers, family, initiative_images=24)["id"]
    conversations.start_autonomy(tasked, "a witch with clean hands", 40, 2.0)
    with patch("app.conversation_turn.stream_chat", Script(("Sigo con ello.", []))):
        _send(api_client, auth_headers, tasked, "más niebla")
    # Its reply launched nothing, and nothing is pending: the task is settled as done, not replaced.
    task = conversations.get(tasked)["autonomy"]
    assert (task["kind"], task["goal"], task["max_images"]) == ("goal", "a witch with clean hands", 40)

    ending = _conversation(api_client, auth_headers, family, initiative_images=24)["id"]
    with patch("app.conversation_turn.stream_chat", Script(("Va:", [_launch(1)]))):
        _send(api_client, auth_headers, ending, "hazme una bruja")
    assert conversations.autonomy(conversations.get(ending))["kind"] == "initiative"
    ended = api_client.post(f"/api/conversations/{ending}/finish", auth=auth_headers)
    assert ended.status_code == 200, ended.text
    stored = conversations.get(ending)
    assert stored["finished_at"] and stored["autonomy"]["end_reason"] == "user"


def test_the_budget_is_a_setting_of_each_conversation(api_client, auth_headers, tmp_path):
    family = _family(tmp_path)
    created = api_client.post("/api/conversations", auth=auth_headers,
                              json={"family_id": family["id"], "llm_model": "vendor/model"}).json()
    assert created["initiative_images"] == conversations.INITIATIVE_IMAGES == 24
    changed = api_client.patch(f"/api/conversations/{created['id']}", auth=auth_headers,
                               json={"initiative_images": 60}).json()
    assert changed["initiative_images"] == 60
    too_many = api_client.patch(f"/api/conversations/{created['id']}", auth=auth_headers,
                                json={"initiative_images": 500})
    assert too_many.status_code == 422
    # Conversations from before the setting get the default.
    assert conversations.initiative({"messages": []}) == 24


def test_of_a_big_batch_only_the_first_images_go_ahead_of_other_conversations(api_client, auth_headers, tmp_path,
                                                                             fake_comfy):
    """In comfy-gateway's queue a project's jobs of one level go in the order sent: an exploring batch
    at high would hold up a couple of images another conversation asked for."""
    workflows.ensure_seeded()
    conversation_id = _conversation(api_client, auth_headers, _family(tmp_path), initiative_images=0)["id"]
    script = Script(("Exploro tres direcciones:", [_launch(2), _launch(2, "call-2", "a witch at noon"),
                                                    _launch(1, "call-3", "a witch at night")]))
    with patch("app.conversation_turn.stream_chat", script):
        _send(api_client, auth_headers, conversation_id, "explora")
    assert fake_comfy.priorities == ["high", "high", "normal", "normal", "normal"]

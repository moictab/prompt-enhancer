"""Working on its own (keep_going): the bot launches, looks at the images and launches again by
itself, until the goal is met or its budget -- images made, dollars spent on its replies -- runs out."""
import json
from unittest.mock import patch

from app import conversations, families, runs, workflows
from app.openrouter_client import StreamEnd

from .conftest import message_text


class Script:
    """A fake stream_chat playing one scripted (text, tool_calls) per request, each costing `cost`."""

    def __init__(self, *replies, cost=0.01):
        self.replies = list(replies)
        self.cost = cost
        self.calls = []

    def __call__(self, api_key, model, messages, temperature, context=None, tools=None, tool_choice=None):
        self.calls.append({"messages": messages, "context": context, "tool_choice": tool_choice})
        text, tool_calls = self.replies.pop(0)
        if text:
            yield text
        yield StreamEnd(cost=self.cost, tool_calls=tool_calls)


def _call(name, call_id, **arguments):
    return {"id": call_id, "name": name, "arguments": json.dumps(arguments)}


def _run(call_id, count=1):
    return _call("run_workflow", call_id, workflow="krea-2/txt2img", changes={"positive": "a witch"}, count=count)


def _setup(api_client, auth_headers, tmp_path):
    workflows.ensure_seeded()
    family = families.create_family("Krea 2", "rules", False, path=str(tmp_path / "families.json"))
    return api_client.post("/api/conversations", auth=auth_headers,
                           json={"family_id": family["id"], "llm_model": "vendor/model"}).json()["id"]


def _send(api_client, auth_headers, conversation_id, content):
    response = api_client.post(f"/api/conversations/{conversation_id}/messages", data={"content": content},
                               auth=auth_headers)
    return [json.loads(line) for line in response.text.splitlines() if line.strip()]


def _finish_all(fake_comfy, *prompt_ids):
    for prompt_id in prompt_ids:
        fake_comfy.finish(prompt_id, f"{prompt_id}.png")
    runs.tracker.poll_once()
    runs.wait_until_idle()


def test_the_bot_iterates_on_its_own_until_the_images_run_out(api_client, auth_headers, tmp_path, fake_comfy):
    conversation_id = _setup(api_client, auth_headers, tmp_path)
    script = Script(
        # The user's turn: it takes the goal and the limits, and launches (which ends the turn).
        ("Voy probando.", [_call("keep_going", "k1", goal="a witch with clean hands", max_images=3, max_cost=0.5),
                           _run("r1", 2)]),
        # Its look at the first run: not there yet, so it launches again by itself.
        ("<review image=\"1\" verdict=\"flawed\">hands</review>Las manos fallan; bajo el LoRA.", [_run("r2")]),
        # Its look at the second: no images left in the budget, so it sums up.
        ("La mejor es #3.", []),
    )

    with patch("app.conversation_turn.stream_chat", script):
        _send(api_client, auth_headers, conversation_id, "Itera tú solo hasta que salga bien, máximo 3 imágenes")
        working = conversations.autonomy(conversations.get(conversation_id))
        assert (working["goal"], working["images"], working["max_images"]) == ("a witch with clean hands", 2, 3)
        listed = api_client.get("/api/conversations", auth=auth_headers).json()[0]
        assert listed["autonomy"] == {"images": 2, "max_images": 3, "cost": 0.01, "max_cost": 0.5}

        _finish_all(fake_comfy, "prompt-1", "prompt-2")
        _finish_all(fake_comfy, "prompt-3")

    assert len(script.calls) == 3  # one request per turn: each launch ended its turn
    look, last_look = script.calls[1:]
    # Working on its own, its look at the images may launch (tools on), and it is told so.
    assert look["context"]["automatic"] and look["tool_choice"] is None
    assert "Goal: a witch with clean hands" in message_text(look["messages"][-1])
    assert "launch the next run in this same reply" in message_text(look["messages"][-1])
    # With the images spent, it can only sum up.
    assert last_look["tool_choice"] == "none"
    assert "The budget is spent" in message_text(last_look["messages"][-1])

    stored = conversations.get(conversation_id)
    assert len(stored["runs"]) == 2 and len(fake_comfy.submitted) == 3
    ended = stored["autonomy"]
    assert (ended["active"], ended["end_reason"], ended["images"]) == (False, "images", 3)
    assert ended["cost"] == 0.03  # three requests of 0.01
    assert stored["messages"][-1]["content"] == "La mejor es #3."
    assert api_client.get("/api/conversations", auth=auth_headers).json()[0]["autonomy"] is None


def test_a_launch_must_fit_the_images_left(api_client, auth_headers, tmp_path, fake_comfy):
    conversation_id = _setup(api_client, auth_headers, tmp_path)
    script = Script(("", [_call("keep_going", "k1", goal="g", max_images=2), _run("r1", 4)]), ("Vaya.", []))

    with patch("app.conversation_turn.stream_chat", script):
        _send(api_client, auth_headers, conversation_id, "Sigue tú, máximo 2 imágenes")

    result = json.loads(script.calls[1]["messages"][-1]["content"])
    assert result["error"] == "Only 2 images left in the budget: launch at most that."
    assert fake_comfy.submitted == []
    # Nothing under way: it is over.
    assert conversations.get(conversation_id)["autonomy"]["end_reason"] == "done"


def test_the_money_runs_out_before_another_look_would_fit(api_client, auth_headers, tmp_path, fake_comfy):
    conversation_id = _setup(api_client, auth_headers, tmp_path)
    script = Script(
        ("", [_call("keep_going", "k1", goal="g", max_cost=0.05), _run("r1")]),  # 0.02 spent
        ("Otra.", [_run("r2")]),  # 0.04 spent; another look (0.02) would pass 0.05
        ("Hasta aquí llega el presupuesto.", []),
        cost=0.02,
    )

    with patch("app.conversation_turn.stream_chat", script):
        _send(api_client, auth_headers, conversation_id, "Itera tú, como mucho 5 céntimos")
        _finish_all(fake_comfy, "prompt-1")
        _finish_all(fake_comfy, "prompt-2")

    assert script.calls[-1]["tool_choice"] == "none"
    ended = conversations.get(conversation_id)["autonomy"]
    assert (ended["active"], ended["end_reason"]) == (False, "cost")


def test_it_ends_when_the_bot_stops_launching(api_client, auth_headers, tmp_path, fake_comfy):
    conversation_id = _setup(api_client, auth_headers, tmp_path)
    script = Script(("", [_call("keep_going", "k1", goal="g"), _run("r1")]),
                    ("Conseguido: #1 está perfecta.", []))

    with patch("app.conversation_turn.stream_chat", script):
        _send(api_client, auth_headers, conversation_id, "Itera tú")
        _finish_all(fake_comfy, "prompt-1")

    assert conversations.get(conversation_id)["autonomy"]["end_reason"] == "done"


def test_working_on_its_own_it_looks_at_the_images_even_with_comments_off(api_client, auth_headers, tmp_path,
                                                                           fake_comfy):
    conversation_id = _setup(api_client, auth_headers, tmp_path)
    api_client.patch(f"/api/conversations/{conversation_id}", auth=auth_headers, json={"auto_comment": False})
    script = Script(("", [_call("keep_going", "k1", goal="g", max_images=1), _run("r1")]), ("Listo: #1.", []))

    with patch("app.conversation_turn.stream_chat", script):
        _send(api_client, auth_headers, conversation_id, "Itera tú")
        _finish_all(fake_comfy, "prompt-1")

    look = script.calls[1]
    assert look["context"]["automatic"]
    assert "unless you're working on your own" not in message_text(look["messages"][-1])  # comments are on now
    assert conversations.get(conversation_id)["messages"][-1]["content"] == "Listo: #1."


def test_the_user_stops_it_or_cancels_a_run(api_client, auth_headers, tmp_path, fake_comfy):
    conversation_id = _setup(api_client, auth_headers, tmp_path)
    url = f"/api/conversations/{conversation_id}"
    script = Script(("", [_call("keep_going", "k1", goal="g"), _run("r1")]),
                    ("", [_call("keep_going", "k2", goal="g2"), _run("r2")]))

    with patch("app.conversation_turn.stream_chat", script):
        _send(api_client, auth_headers, conversation_id, "Itera tú")
        stopped = api_client.post(f"{url}/autonomy/stop", auth=auth_headers)
        assert stopped.status_code == 200 and stopped.json()["end_reason"] == "user"
        assert api_client.post(f"{url}/autonomy/stop", auth=auth_headers).status_code == 409

        _send(api_client, auth_headers, conversation_id, "Vuelve a iterar tú")
        assert conversations.autonomy(conversations.get(conversation_id))["goal"] == "g2"
        assert api_client.post(f"{url}/runs/2/cancel", auth=auth_headers).status_code == 200

    assert conversations.get(conversation_id)["autonomy"]["end_reason"] == "user"

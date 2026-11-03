"""Conversar with ComfyUI: tool calls within a turn, runs followed to their results,
the bot's automatic comment, and the bot's changes to existing notes."""
import json
from unittest.mock import patch

import requests

from app import comfy, conversation_turn, conversations, families, memory, runs, workflows
from app.openrouter_client import StreamEnd

from .conftest import png_bytes

REQUEST_ID = "20261103-102219-0123abcd"


def _family(tmp_path):
    return families.create_family("Krea", "rules", False, path=str(tmp_path / "families.json"))


def _conversation(api_client, auth_headers, family):
    return api_client.post("/api/conversations", auth=auth_headers,
                           json={"family_id": family["id"], "llm_model": "vendor/model"}).json()


def _events(response):
    return [json.loads(line) for line in response.text.splitlines() if line.strip()]


def _send(api_client, auth_headers, conversation_id, content="Hazme una bruja", files=None):
    return api_client.post(f"/api/conversations/{conversation_id}/messages", data={"content": content},
                           files=files, auth=auth_headers)


class Script:
    """A fake stream_chat that plays one scripted reply per request: (text, tool_calls)."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.calls = []

    def __call__(self, api_key, model, messages, temperature, context=None, tools=None, tool_choice=None):
        self.calls.append({"messages": messages, "context": context, "tools": tools, "tool_choice": tool_choice})
        text, tool_calls = self.replies.pop(0)
        if text:
            yield text
        yield StreamEnd(cost=0.01, request_id=f"{REQUEST_ID[:-1]}{len(self.calls)}", tool_calls=tool_calls)


def _call(name, call_id="call-1", **arguments):
    return {"id": call_id, "name": name, "arguments": json.dumps(arguments)}


RUN_CALL = _call("run_workflow", workflow="krea-2/txt2img", changes={"positive": "a witch at dawn"}, count=2)


def test_a_turn_runs_a_workflow_and_keeps_its_steps(api_client, auth_headers, tmp_path, fake_comfy):
    workflows.ensure_seeded()
    conversation = _conversation(api_client, auth_headers, _family(tmp_path))
    script = Script(("Voy con ello.", [RUN_CALL]),
                    ("<title>Bruja</title>Lanzada:\n<prompt>\nPOSITIVE: a witch at dawn\n</prompt>", []))

    with patch("app.conversation_turn.stream_chat", script):
        events = _events(_send(api_client, auth_headers, conversation["id"]))

    assert [e["type"] for e in events] == ["delta", "tool", "delta", "done"]
    assert events[1]["part"] == {"type": "run", "run": 1, "workflow": "krea-2/txt2img", "count": 2}
    first, second = script.calls
    assert first["tools"] is conversation_turn.chat_tools.TOOLS and first["tool_choice"] is None
    assert "## Workflow Library\n- krea-2/txt2img:" in first["messages"][0]["content"]
    assert second["context"]["step"] == 2
    # The second request carries the call and its result.
    assert second["messages"][-2]["tool_calls"][0]["function"]["name"] == "run_workflow"
    assert json.loads(second["messages"][-1]["content"])["run"] == 1

    message = events[-1]["message"]
    assert [p["type"] for p in message["parts"]] == ["text", "run", "text", "prompt"]
    assert message["cost"] == 0.02
    assert message["request_ids"] == [f"{REQUEST_ID[:-1]}1", f"{REQUEST_ID[:-1]}2"]
    assert message["steps"][0]["tool_calls"] == [RUN_CALL]
    assert events[-1]["conversation"]["title"] == "Bruja"
    assert len(fake_comfy.submitted) == 2

    stored = api_client.get(f"/api/conversations/{conversation['id']}", auth=auth_headers).json()
    assert stored["runs"][0]["status"] == "pending"


def test_finished_runs_bring_their_images_and_an_automatic_comment(api_client, auth_headers, tmp_path, fake_comfy):
    workflows.ensure_seeded()
    conversation_id = _conversation(api_client, auth_headers, _family(tmp_path))["id"]
    with patch("app.conversation_turn.stream_chat", Script(("", [RUN_CALL]), ("Lanzada.", []))):
        _send(api_client, auth_headers, conversation_id)

    comment = Script(("La 2 tiene mejor luz.", []))
    with patch("app.conversation_turn.stream_chat", comment):
        fake_comfy.finish("prompt-1", "a.png")
        runs.tracker.poll_once()
        assert conversations.get_run(conversation_id, 1)["status"] == "pending"  # one job still out
        fake_comfy.finish("prompt-2", "b.png")
        runs.tracker.poll_once()
        runs.wait_until_idle()

    stored = api_client.get(f"/api/conversations/{conversation_id}", auth=auth_headers).json()
    run = stored["runs"][0]
    assert run["status"] == "done" and "_images" not in run
    assert [m["role"] for m in stored["messages"]] == ["user", "assistant", "results", "assistant"]
    results, reply = stored["messages"][2:]
    assert results["run"] == 1 and len(results["images"]) == 2
    assert [j["images"] for j in run["jobs"]] == [[results["images"][0]["id"]], [results["images"][1]["id"]]]
    assert reply["automatic"] is True and reply["content"] == "La 2 tiene mejor luz."
    [request] = comment.calls
    assert request["tool_choice"] == "none" and request["context"]["automatic"] is True
    last = request["messages"][-1]["content"]
    assert last[0]["text"].startswith("[Run 1 (krea-2/txt2img) finished: image 1 (seed")
    assert last[0]["text"].endswith(conversation_turn.chat.AUTOMATIC_NOTE)
    assert sum(1 for p in last if p["type"] == "image_url") == 2

    # The image a run made is served, and image:N reruns its exact workflow.
    image_id = results["images"][0]["id"]
    assert api_client.get(f"/api/conversations/{conversation_id}/images/{image_id}",
                          auth=auth_headers).status_code == 200
    assert conversations.image_workflow(conversation_id, image_id) == conversations.run_workflows(conversation_id, 1)[0]
    download = api_client.get(f"/api/conversations/{conversation_id}/runs/1/workflow?job=2", auth=auth_headers)
    assert download.json() == fake_comfy.submitted[1]
    assert "attachment" in download.headers["content-disposition"]


def _launch_two(api_client, auth_headers, tmp_path):
    workflows.ensure_seeded()
    conversation_id = _conversation(api_client, auth_headers, _family(tmp_path))["id"]
    with patch("app.conversation_turn.stream_chat", Script(("", [RUN_CALL]), ("Va.", []))):
        _send(api_client, auth_headers, conversation_id)
    return conversation_id


def test_a_download_that_fails_is_tried_again(api_client, auth_headers, tmp_path, fake_comfy):
    conversation_id = _launch_two(api_client, auth_headers, tmp_path)
    real_view, failures = fake_comfy.view, {"left": 2}

    def stalling(base, filename, subfolder="", type_="output"):
        if failures["left"]:
            failures["left"] -= 1
            raise requests.exceptions.ReadTimeout("ComfyUI stalled")
        return real_view(base, filename, subfolder, type_)

    fake_comfy.view = stalling
    with patch("app.conversation_turn.stream_chat", Script(("Bien.", []))):
        fake_comfy.finish("prompt-1", "out-a")
        fake_comfy.finish("prompt-2", "out-b")
        runs.tracker.poll_once()
        run = conversations.get_run(conversation_id, 1)
        assert run["status"] == "pending"  # nothing given up, nothing half-stored
        assert [j["download_attempts"] for j in run["jobs"]] == [1, 1]
        runs.tracker.poll_once()
        runs.wait_until_idle()

    assert conversations.get_run(conversation_id, 1)["status"] == "done"
    assert len(conversations.get(conversation_id)["messages"][2]["images"]) == 2


def test_lost_downloads_can_be_fetched_again_into_the_same_message(api_client, auth_headers, tmp_path, fake_comfy):
    conversation_id = _launch_two(api_client, auth_headers, tmp_path)
    real_view = fake_comfy.view

    def refused(base, filename, subfolder="", type_="output"):
        raise comfy.ComfyError(403, "refused")

    fake_comfy.view = refused
    with patch("app.conversation_turn.stream_chat", Script(("No han llegado.", []))):
        fake_comfy.finish("prompt-1", "out-a")
        fake_comfy.finish("prompt-2", "out-b")
        for _ in range(runs.MAX_DOWNLOAD_ATTEMPTS):
            runs.tracker.poll_once()
        runs.wait_until_idle()
    run = conversations.get_run(conversation_id, 1)
    assert run["status"] == "error"
    assert all(j["error"] == "Couldn't download out-a" for j in run["jobs"][:1])
    assert [m["role"] for m in conversations.get(conversation_id)["messages"]] == [
        "user", "assistant", "results", "assistant"]

    fake_comfy.view = real_view
    retry = f"/api/conversations/{conversation_id}/runs/1/retry-downloads"
    assert api_client.post(retry, auth=auth_headers).json()["status"] == "pending"
    with patch("app.conversation_turn.stream_chat", Script()) as idle:
        runs.tracker.poll_once()
        runs.wait_until_idle()

    conversation = conversations.get(conversation_id)
    # The images land in the results message already in the chat; no new message, no new comment.
    assert [m["role"] for m in conversation["messages"]] == ["user", "assistant", "results", "assistant"]
    assert len(conversation["messages"][2]["images"]) == 2
    assert conversations.get_run(conversation_id, 1)["status"] == "done"
    assert idle.calls == []
    assert api_client.post(retry, auth=auth_headers).status_code == 409


def test_a_failed_run_is_reported_and_commented(api_client, auth_headers, tmp_path, fake_comfy):
    workflows.ensure_seeded()
    conversation_id = _conversation(api_client, auth_headers, _family(tmp_path))["id"]
    call = _call("run_workflow", workflow="krea-2/txt2img")
    with patch("app.conversation_turn.stream_chat", Script(("", [call]), ("Va.", []))):
        _send(api_client, auth_headers, conversation_id)

    comment = Script(("Se quedó sin memoria; bajo la resolución.", []))
    with patch("app.conversation_turn.stream_chat", comment):
        fake_comfy.finish("prompt-1", status="error", error="CUDA out of memory")
        runs.tracker.poll_once()
        runs.wait_until_idle()

    run = conversations.get_run(conversation_id, 1)
    assert run["status"] == "error"
    assert run["jobs"][0]["error"] == "RuntimeError: CUDA out of memory (in KSampler)"
    assert "1 of 1 job failed: RuntimeError: CUDA out of memory" in comment.calls[0]["messages"][-1]["content"]


def test_cancelled_runs_get_no_comment(api_client, auth_headers, tmp_path, fake_comfy):
    workflows.ensure_seeded()
    conversation_id = _conversation(api_client, auth_headers, _family(tmp_path))["id"]
    with patch("app.conversation_turn.stream_chat", Script(("", [RUN_CALL]), ("Va.", []))):
        _send(api_client, auth_headers, conversation_id)

    with patch("app.conversation_turn.stream_chat", Script()) as idle:
        response = api_client.post(f"/api/conversations/{conversation_id}/runs/1/cancel", auth=auth_headers)
        runs.wait_until_idle()

    assert response.json()["status"] == "cancelled"
    assert fake_comfy.cancelled == ["prompt-1", "prompt-2"]
    assert idle.calls == []
    stored = conversations.get(conversation_id)
    assert [m["role"] for m in stored["messages"]] == ["user", "assistant"]
    assert api_client.post(f"/api/conversations/{conversation_id}/runs/1/cancel",
                           auth=auth_headers).status_code == 409
    assert api_client.post(f"/api/conversations/{conversation_id}/runs/7/cancel",
                           auth=auth_headers).status_code == 404


def test_no_automatic_comment_once_the_user_has_written(api_client, auth_headers, tmp_path, fake_comfy):
    family = _family(tmp_path)
    conversation_id = _conversation(api_client, auth_headers, family)["id"]
    conversations.append_message(conversation_id, {"role": "results", "run": 1, "content": "", "images": []})
    conversations.append_message(conversation_id, {"role": "user", "content": "¿y bien?", "images": []})

    assert list(conversation_turn.run_turn(conversation_id, automatic=True)) == []


def test_the_last_step_offers_no_more_tools(api_client, auth_headers, tmp_path, fake_comfy):
    workflows.ensure_seeded()
    conversation_id = _conversation(api_client, auth_headers, _family(tmp_path))["id"]
    inspect = _call("inspect_workflow", workflow="krea-2/txt2img")
    script = Script(*[("", [inspect])] * (conversation_turn.MAX_STEPS - 1), ("Ya está.", []))

    with patch("app.conversation_turn.stream_chat", script):
        events = _events(_send(api_client, auth_headers, conversation_id))

    assert len(script.calls) == conversation_turn.MAX_STEPS
    assert [c["tool_choice"] for c in script.calls] == [None] * (conversation_turn.MAX_STEPS - 1) + ["none"]
    assert events[-1]["type"] == "done"


def test_the_bot_rewrites_and_forgets_existing_notes(api_client, auth_headers, tmp_path):
    first = memory.create_note("Le gusta mucho la luz fría del amanecer en exteriores")
    second = memory.create_note("Usa SDXL")
    third = memory.create_note("Odia el grano")
    conversation_id = _conversation(api_client, auth_headers, _family(tmp_path))["id"]
    script = Script(('Hecho. <remember id="1">likes: cool dawn light</remember><forget id="3"/>'
                     "<remember>krea-2: ink LoRA &lt;0.3</remember>", []))

    with patch("app.conversation_turn.stream_chat", script):
        message = _events(_send(api_client, auth_headers, conversation_id))[-1]["message"]

    assert "[1] Le gusta mucho la luz fría" in script.calls[0]["messages"][0]["content"]
    notes = {n["id"]: n["text"] for n in memory.list_notes()}
    assert notes[first["id"]] == "likes: cool dawn light"
    assert notes[second["id"]] == "Usa SDXL"
    assert third["id"] not in notes
    assert message["note_changes"] == [
        {"action": "update", "id": first["id"], "text": "likes: cool dawn light",
         "previous": "Le gusta mucho la luz fría del amanecer en exteriores"},
        {"action": "forget", "id": third["id"], "text": "Odia el grano"},
    ]
    assert [n["text"] for n in message["notes"]] == ["krea-2: ink LoRA &lt;0.3"]
    assert message["content"] == "Hecho."


def test_shared_images_keep_their_metadata_and_a_small_copy(api_client, auth_headers, tmp_path):
    conversation_id = _conversation(api_client, auth_headers, _family(tmp_path))["id"]
    graph = {"1": {"class_type": "KSampler", "inputs": {"seed": 5, "positive": ["2", 0]}},
             "2": {"class_type": "CLIPTextEncode", "inputs": {"text": "a fox"}}}
    big = png_bytes(size=(2400, 1600), prompt=graph)

    with patch("app.conversation_turn.stream_chat", Script(("Bonito zorro.", []))) as script:
        _send(api_client, auth_headers, conversation_id, files=[("images", ("fox.png", big, "image/png"))])

    [image] = conversations.get(conversation_id)["messages"][0]["images"]
    assert image["small"] is True
    assert image["metadata"]["positive"] == "a fox"
    assert conversations.image_workflow(conversation_id, image["id"]) == graph
    small = api_client.get(f"/api/conversations/{conversation_id}/images/{image['id']}?small=true", auth=auth_headers)
    assert small.headers["content-type"] == "image/jpeg"
    full = api_client.get(f"/api/conversations/{conversation_id}/images/{image['id']}", auth=auth_headers)
    assert full.content == big
    sent = script.calls[0]["messages"][1]["content"]
    assert "Image 1 metadata (PNG, 2400x1600, made with ComfyUI" in sent[1]["text"]
    assert sent[2]["image_url"]["url"].startswith("data:image/jpeg;base64,")


def test_reactions_travel_with_the_next_message(api_client, auth_headers, tmp_path):
    conversation_id = _conversation(api_client, auth_headers, _family(tmp_path))["id"]
    images = [conversations.save_image(conversation_id, png_bytes(), "image/png") for _ in range(2)]
    conversations.append_message(conversation_id, {"role": "results", "run": 1, "content": "", "images": images})
    put = f"/api/conversations/{conversation_id}/images"

    assert api_client.put(f"{put}/{images[0]['id']}/reaction", auth=auth_headers,
                          json={"rating": "up"}).json()["reaction"]["rating"] == "up"
    api_client.put(f"{put}/{images[1]['id']}/reaction", auth=auth_headers,
                   json={"rating": "down", "comment": " hands  wrong "})
    assert api_client.put(f"{put}/{'0' * 32}/reaction", auth=auth_headers, json={"rating": "up"}).status_code == 404
    assert api_client.put(f"{put}/{images[0]['id']}/reaction", auth=auth_headers,
                          json={"rating": "meh"}).status_code == 422

    # Reactions alone are enough to send.
    script = Script(("Arreglo las manos.", []))
    with patch("app.conversation_turn.stream_chat", script):
        response = _send(api_client, auth_headers, conversation_id, content="")

    assert response.status_code == 200
    user = conversations.get(conversation_id)["messages"][1]
    assert user["reactions"] == [
        {"image_id": images[0]["id"], "rating": "up", "comment": ""},
        {"image_id": images[1]["id"], "rating": "down", "comment": "hands wrong"},
    ]
    # The results and the user's message are one user turn; the reactions close it.
    sent = script.calls[0]["messages"][-1]["content"]
    assert [p["text"] for p in sent if p["type"] == "text"][-1] == (
        '[Reactions: image 1 liked; image 2 disliked: "hands wrong"]')

    # Sent once: the next message doesn't carry them again, and nothing is left to send.
    assert _send(api_client, auth_headers, conversation_id, content="").status_code == 400
    # Clearing a reaction removes it.
    cleared = api_client.put(f"{put}/{images[0]['id']}/reaction", auth=auth_headers, json={"rating": None})
    assert "reaction" not in cleared.json()


def test_anthropic_models_get_cache_breakpoints():
    messages = [{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "x", "tool_calls": []}, {"role": "tool", "content": "{}"}]

    marked = conversation_turn.with_cache_breakpoints("anthropic/claude-sonnet-4.6", messages, history_length=2)

    assert marked[0]["content"] == [{"type": "text", "text": "sys", "cache_control": {"type": "ephemeral"}}]
    assert marked[1]["content"] == [{"type": "text", "text": "hi", "cache_control": {"type": "ephemeral"}}]
    assert marked[2:] == messages[2:]
    assert messages[0]["content"] == "sys"  # the original is untouched
    assert conversation_turn.with_cache_breakpoints("openai/gpt-5", messages, 2) is messages


def test_deleting_a_conversation_cancels_its_pending_runs(api_client, auth_headers, tmp_path, fake_comfy):
    workflows.ensure_seeded()
    conversation_id = _conversation(api_client, auth_headers, _family(tmp_path))["id"]
    with patch("app.conversation_turn.stream_chat", Script(("", [RUN_CALL]), ("Va.", []))):
        _send(api_client, auth_headers, conversation_id)

    assert api_client.delete(f"/api/conversations/{conversation_id}", auth=auth_headers).status_code == 200
    runs.wait_until_idle()

    assert fake_comfy.cancelled == ["prompt-1", "prompt-2"]
    assert conversations.get(conversation_id) is None


def test_the_comment_follows_the_results_within_one_hold_of_the_lock(api_client, auth_headers, tmp_path, fake_comfy):
    """While the bot comments, the conversation says so: the chat keeps checking back."""
    workflows.ensure_seeded()
    conversation_id = _conversation(api_client, auth_headers, _family(tmp_path))["id"]
    with patch("app.conversation_turn.stream_chat", Script(("", [RUN_CALL]), ("Va.", []))):
        _send(api_client, auth_headers, conversation_id)
    seen = []

    def comment(api_key, model, messages, temperature, context=None, tools=None, tool_choice=None):
        seen.append(conversation_turn.is_commenting(conversation_id))
        yield "Bien."
        yield StreamEnd(cost=None)

    with patch("app.conversation_turn.stream_chat", comment):
        fake_comfy.finish("prompt-1", "a.png")
        fake_comfy.finish("prompt-2", "b.png")
        runs.tracker.poll_once()
        runs.wait_until_idle()

    assert seen == [True]
    assert not conversation_turn.is_commenting(conversation_id)
    assert [m["role"] for m in conversations.get(conversation_id)["messages"]][-2:] == ["results", "assistant"]


def test_a_first_run_names_its_folder_after_the_title_being_written(api_client, auth_headers, tmp_path, fake_comfy):
    workflows.ensure_seeded()
    conversation_id = _conversation(api_client, auth_headers, _family(tmp_path))["id"]
    script = Script(("<title>Bruja al alba</title>Voy.", [RUN_CALL]), ("Lanzada.", []))

    with patch("app.conversation_turn.stream_chat", script):
        _send(api_client, auth_headers, conversation_id)

    folder = conversations.get(conversation_id)["output_folder"]
    assert folder.endswith(f"_bruja-al-alba_{conversation_id[:8]}")
    assert conversations.get_run(conversation_id, 1)["output_folder"] == f"prompt-enhancer/{folder}"

"""Ending a conversation, and taking stock of it first: the app hands the bot every image its runs
made, with the recipe, its review and the user's reaction, and the bot looks back and fixes its notes."""
import uuid
from unittest.mock import patch

from app import balance, conversations, memory, runs, workflows

from .conftest import message_text
from .test_api_conversar_comfy import Script, _call, _events, _family, _send

MODEL = "krea-2\\krea-2-turbo-official-comfy-int8-convrot_v1.0.safetensors"
LAUNCH = _call("run_workflow", workflow="krea-2/txt2img", count=2, changes={
    "positive": "a witch at dawn", "loras": [{"name": "style-ink_v1", "strength": 0.6}]})


def _two_images(api_client, auth_headers, tmp_path, fake_comfy):
    """A conversation whose one run made two images, without automatic comments."""
    workflows.ensure_seeded()
    fake_comfy.model_files["diffusion_models"] = [MODEL]
    conversation_id = api_client.post("/api/conversations", auth=auth_headers, json={
        "family_id": _family(tmp_path)["id"], "llm_model": "vendor/model", "auto_comment": False}).json()["id"]
    reply = f'Va:\n<prompt model="{MODEL}">\nPOSITIVE: a witch at dawn\n</prompt>'
    with patch("app.conversation_turn.stream_chat", Script((reply, [LAUNCH]))):
        _send(api_client, auth_headers, conversation_id)
    fake_comfy.finish("prompt-1", "a.png")
    fake_comfy.finish("prompt-2", "b.png")
    runs.tracker.poll_once()
    runs.wait_until_idle()
    return conversation_id


def _image_ids(conversation_id):
    return [img["id"] for m in conversations.get(conversation_id)["messages"] for img in m.get("images") or []]


def _react(api_client, auth_headers, conversation_id, image_id, rating, comment=""):
    response = api_client.put(f"/api/conversations/{conversation_id}/images/{image_id}/reaction",
                              auth=auth_headers, json={"rating": rating, "comment": comment})
    assert response.status_code == 200, response.text


def test_taking_stock_hands_the_bot_the_whole_record_and_its_notes_follow(api_client, auth_headers, tmp_path,
                                                                          fake_comfy):
    conversation_id = _two_images(api_client, auth_headers, tmp_path, fake_comfy)
    first, second = _image_ids(conversation_id)
    conversations.set_review(conversation_id, second, "flawed", "hands melted")
    _react(api_client, auth_headers, conversation_id, first, "up")
    _react(api_client, auth_headers, conversation_id, second, "down", "manos rotas")
    memory.create_note("krea-2: style-ink burns faces")

    looked_back = Script(('Lo que mejor ha ido: #1, con style-ink a 0.6.\n'
                          '<remember id="1">krea-2: style-ink fine at 0.6, hands may break</remember>', []))
    with patch("app.conversation_turn.stream_chat", looked_back):
        response = api_client.post(f"/api/conversations/{conversation_id}/finish", auth=auth_headers,
                                   data={"focus": "sobre todo las manos"})
    events = _events(response)
    assert events[-1]["type"] == "done", events[-1]

    # One request, with the tools (the cached prefix stays the same) but none to call.
    [request] = looked_back.calls
    assert request["tools"] and request["tool_choice"] == "none"
    asked = message_text(request["messages"][-1])  # after the run's results, which waited for it
    assert asked.startswith("[Run 1 (krea-2/txt2img")
    assert "[The user is ending this conversation and asked you to take stock of it first." in asked
    assert "2 images from 1 runs: the user liked 1, disliked 1 and didn't react to 0; you reviewed 1 (1 flawed)." in asked
    assert ("Run 1: workflow krea-2/txt2img, proposal 1; krea-2\\krea-2-turbo-official-comfy-int8-convrot_v1.0; "
            "LoRAs style-ink_v1 0.6;") in asked
    assert "- Image 1 (seed " in asked and "): user liked; you: not reviewed." in asked
    assert '): user disliked: "manos rotas"; you: flawed: "hands melted".' in asked
    assert "Base model: krea-2\\krea-2-turbo-official-comfy-int8-convrot_v1.0 in every image." in asked
    assert "LoRA and weight: style-ink_v1 at 0.6 in every image." in asked
    assert "bring your notes in line with this evidence" in asked
    assert "The user adds: sobre todo las manos" in asked
    assert '[Reactions: image 1 liked; image 2 disliked: "manos rotas"]' in asked  # the unsent reactions go with it
    assert "[1] krea-2: style-ink burns faces" in asked  # the notes to bring in line

    # The note it rewrote, with the previous text kept for "Deshacer".
    assert [n["text"] for n in memory.list_notes()] == ["krea-2: style-ink fine at 0.6, hands may break"]
    assert events[-1]["message"]["note_changes"][0]["previous"] == "krea-2: style-ink burns faces"

    stored = conversations.get(conversation_id)["messages"][-2]
    assert stored["role"] == "user" and stored["content"] == "sobre todo las manos"
    assert stored["balance_counts"] == {"images": 2, "runs": 1, "liked": 1, "loved": 0, "disliked": 1, "unrated": 0,
                                        "reviewed": 1, "flawed": 1}
    # The chat gets the request without the evidence; GET …/balance has it.
    shown = api_client.get(f"/api/conversations/{conversation_id}", auth=auth_headers).json()["messages"][-2]
    assert shown["balance"] is True and shown["balance_counts"]["images"] == 2
    data = api_client.get(f"/api/conversations/{conversation_id}/balance", auth=auth_headers).json()
    assert data["images"] == 2 and data["unrated"] == 0 and "## Counts per ingredient" in data["text"]

    assert stored["balance"] == asked[asked.index("[The user is ending"):asked.index("\n\nThe user adds:")]
    listed = next(s for s in api_client.get("/api/conversations", auth=auth_headers).json() if s["id"] == conversation_id)
    assert listed["finished_at"]

    # Writing again reopens it; the evidence, answered, comes back as one line (tens of KB less every turn).
    with patch("app.conversation_turn.stream_chat", Script(("Vale.", []))) as later:
        _send(api_client, auth_headers, conversation_id, "sigamos")
    assert "finished_at" not in conversations.get(conversation_id)
    replayed = "\n".join(message_text(m) for m in later.calls[0]["messages"] if m["role"] == "user")
    assert stored["balance"] not in replayed and "## Every image, by run" not in replayed
    assert ("[The user ended the conversation and asked you to take stock of it: the app sent every image's recipe, "
            "review and reaction (2 images; 1 liked, 1 disliked, 0 unrated). Your balance follows.]\n\n"
            "The user adds: sobre todo las manos") in replayed


def test_without_images_ending_only_marks_it_and_not_while_it_works_alone(api_client, auth_headers, tmp_path,
                                                                           fake_comfy):
    workflows.ensure_seeded()
    conversation_id = api_client.post("/api/conversations", auth=auth_headers, json={
        "family_id": _family(tmp_path)["id"], "llm_model": "vendor/model"}).json()["id"]
    with patch("app.conversation_turn.stream_chat", Script(("Hola.", []))):
        _send(api_client, auth_headers, conversation_id, "hola")
    with patch("app.conversation_turn.stream_chat", Script()) as nothing:
        ended = api_client.post(f"/api/conversations/{conversation_id}/finish", auth=auth_headers)
    assert ended.status_code == 200 and ended.json()["finished_at"] and nothing.calls == []
    assert [m["role"] for m in conversations.get(conversation_id)["messages"]] == ["user", "assistant"]
    assert api_client.get(f"/api/conversations/{conversation_id}/balance", auth=auth_headers).status_code == 404

    working = _two_images(api_client, auth_headers, tmp_path, fake_comfy)
    conversations.start_autonomy(working, "a clean witch", 8, 1.0)
    busy = api_client.post(f"/api/conversations/{working}/finish", auth=auth_headers)
    assert busy.status_code == 409 and "working on its own" in busy.json()["detail"]
    assert conversations.get(working)["messages"][-1]["role"] == "results"  # nothing stored


def _job(seed, images, loras=(), weight=None):
    job = {"seed": seed, "images": images, "status": "done", "models": [MODEL],
           "loras": [{"name": name, "strength": strength} for name, strength in loras]}
    if weight is not None:
        job["calibration_weight"] = weight
    return job


def test_the_record_spells_each_recipe_once_and_counts_side_by_side():
    images = [{"id": f"img-{i}"} for i in range(1, 6)]
    images[0]["reaction"] = {"rating": "up"}
    images[4]["review"] = {"verdict": "flawed", "note": "burnt"}
    conversation = {
        "id": str(uuid.uuid4()),
        "messages": [
            {"role": "assistant", "content": "<prompt>\nPOSITIVE: a witch\n</prompt>\n<prompt>\nPOSITIVE: a cat\n</prompt>"},
            {"role": "results", "run": 1, "images": images[:2]},
            {"role": "results", "run": 2, "images": images[2:3]},
            {"role": "results", "run": 3, "images": images[3:5]},
        ],
        "runs": [
            {"n": 1, "workflow": "krea-2/txt2img", "changes": {"positive": "A cat "}, "size": "1:1 (Square), 1.0 MP",
             "base_models": [MODEL], "jobs": [_job(1, ["img-1"], [("krea-2\\ink.safetensors", 0.6)]),
                                              _job(2, ["img-2"], [("krea-2\\ink.safetensors", 0.6)])]},
            {"n": 2, "workflow": "run:1", "changes": {"positive": "a dog"}, "size": "1:1 square, 1 MP",
             "base_models": [MODEL], "jobs": [_job(3, ["img-3"], [("krea-2\\ink.safetensors", 0.6)])]},
            {"n": 3, "workflow": "krea-2/txt2img", "kind": "calibration", "lora": "krea-2\\ink.safetensors",
             "changes": {"positive": "a witch"}, "size": "1:1, 1 MP", "base_models": [MODEL],
             "jobs": [_job(4, ["img-4"], [("krea-2\\ink.safetensors", 0.4)], weight=0.4),
                      _job(4, ["img-5"], [("krea-2\\ink.safetensors", 1.0)], weight=1.0)]},
            {"n": 4, "workflow": "krea-2/txt2img", "status": "cancelled", "jobs": [{"seed": 5, "images": []}]},
        ],
    }

    found = balance.ledger(conversation)
    text = found.text

    assert found.counts == {"images": 5, "runs": 3, "liked": 1, "loved": 0, "disliked": 0, "unrated": 4, "reviewed": 1,
                            "flawed": 1}
    # The prompt a run ran, matched to the proposal that has it (ignoring case and spacing).
    assert "Run 1: workflow krea-2/txt2img, proposal 2; " in text
    assert "Run 2: workflow run:1, a prompt not among your proposals; same settings as run 1" in text
    assert "Run 3: calibration of ink, workflow krea-2/txt2img, proposal 1; " in text
    assert "- Image 4 (weight 0.4, seed 4): user no reaction; you: not reviewed." in text
    assert "Run 4" not in text  # it made nothing
    # One spelling for the same proportions, whichever way the run stored them.
    assert "Size: 1:1, 1 MP in every image." in text
    # A LoRA's weights side by side, calibration weights included.
    lines = text.splitlines()
    start = lines.index("LoRA and weight:")
    assert lines[start + 1:start + 4] == [
        "- ink at 0.4: 1 image: 0 liked, 0 disliked, 1 unrated",
        "- ink at 0.6: 3 images: 1 liked, 0 disliked, 2 unrated",
        "- ink at 1: 1 image: 0 liked, 0 disliked, 1 unrated; 1 flawed in your reviews",
    ]
    assert balance.ledger({"id": str(uuid.uuid4()), "messages": [], "runs": []}) is None

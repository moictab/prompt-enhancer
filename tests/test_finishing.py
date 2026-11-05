"""Drafts and finishing: how long workflows take, the queue, and finishing a chosen image from itself."""
import json

import pytest

from app import chat, chat_tools, comfy, conversations, image_check, runs, timing, workflows
from app.comfy_graph import Graph

from .conftest import png_bytes


@pytest.fixture
def finishing(fake_comfy):
    """A conversation with a run image made by krea-2/txt2img (a prompt and a LoRA in its recipe),
    and a finishing workflow for krea-2 in the library (it takes an image)."""
    workflows.ensure_seeded()
    recipe = workflows.find_by_name("krea-2/txt2img")["workflow"]
    made = Graph(json.loads(json.dumps(recipe)))
    made.set_prompt(positive="a witch at dusk")
    made.set_loras([{"name": "krea-2\\style-ink_v1.safetensors", "strength": 0.4}])
    finish = Graph(json.loads(json.dumps(recipe)))
    finish.nodes["90"] = {"class_type": "LoadImage", "inputs": {"image": "(the image)"}}
    finish.nodes["91"] = {"class_type": "VAEEncode", "inputs": {"pixels": ["90", 0], "vae": ["139", 0]}}
    finish.nodes["53"]["inputs"].update(latent_image=["91", 0], denoise=0.35)
    workflows.create("krea-2/finish-image", "Finishes an image: upscale and detail.", finish.to_api())
    conversation = conversations.create("f", "m", 0.7)
    image = conversations.save_image(conversation["id"], png_bytes(), "image/png", workflow=made.to_api())
    conversations.append_message(conversation["id"], {"role": "user", "content": "una bruja", "images": []})
    conversations.append_message(conversation["id"], {"role": "results", "run": 99, "content": "", "images": [image]})
    return chat_tools.ToolBox(conversation["id"], fake_comfy), fake_comfy, conversation["id"], image["id"]


def test_finishing_an_image_runs_the_familys_finishing_workflow_on_it_with_its_recipe(finishing):
    box, fake, conversation_id, image_id = finishing

    result, part = box.run("finish_image", json.dumps({"image": 1}))

    assert "error" not in result, result
    assert part == {"type": "run", "run": 1, "workflow": "krea-2/finish-image", "count": 1, "finish_of": 1}
    [sent] = fake.submitted
    assert sent["90"]["inputs"]["image"] == f"{comfy.UPLOAD_SUBFOLDER}/{conversation_id}-{image_id}.png"
    assert sent["63"]["inputs"]["value"] == "a witch at dusk"
    assert [n["inputs"]["lora_name"] for n in sent.values() if n["class_type"] == "LoraLoader"] == [
        "krea-2\\style-ink_v1.safetensors"]
    assert sent["53"]["inputs"]["denoise"] == 0.35
    run = conversations.get_run(conversation_id, 1)
    assert (run["kind"], run["finish_of"]) == ("finish", 1)
    assert chat.results_text(run, {}, automatic=False).startswith(
        "[Run 1 (finishing image 1 with krea-2/finish-image) produced no images.")

    # A tweak goes on top of the recipe.
    box.run("finish_image", json.dumps({"image": 1, "changes": {"loras": []}}))
    assert not [n for n in fake.submitted[1].values() if n["class_type"] == "LoraLoader"]


def test_finishing_waits_behind_the_drafts_and_leaves_them_their_place(finishing):
    """A finish is minutes of GPU: at "high" it spent the hour's budget the drafts the user waits for
    need, and at "low" the gateway would stop it for them and start it over. So "normal", whoever
    asked, and a draft launched after it in the same reply still goes at "high"."""
    box, fake, conversation_id, image_id = finishing

    box.run("finish_image", json.dumps({"image": 1}))
    box.run("run_workflow", json.dumps({"workflow": "krea-2/txt2img", "changes": {"positive": "a witch"}}))
    box.automatic = True
    box.run("finish_image", json.dumps({"image": 1}))

    assert fake.priorities == ["normal", "high", "normal"]


def test_without_a_finishing_workflow_the_bot_is_told_what_to_do(finishing):
    box, fake, _, _ = finishing
    workflows.delete(workflows.find_by_name("krea-2/finish-image")["id"])

    result, part = box.run("finish_image", json.dumps({"image": 1}))

    assert 'none named "krea-2/finish-image"' in result["error"] and part["error"] and not fake.submitted
    assert "There is no image 9" in box.run("finish_image", json.dumps({"image": 9}))[0]["error"]


def test_a_finished_image_is_inspected_as_finished(finishing, monkeypatch):
    box, fake, conversation_id, _ = finishing
    box.run("finish_image", json.dumps({"image": 1}))
    finished = conversations.save_image(conversation_id, png_bytes(), "image/png")
    conversations.append_message(conversation_id, {"role": "results", "run": 1, "content": "", "images": [finished]})
    conversation = conversations.get(conversation_id)
    conversation["engine"] = "claude-code"
    monkeypatch.setattr(image_check, "ACTIVE", True)  # off app-wide for now
    asked = []
    monkeypatch.setattr("app.claude_code.ask", lambda system, prompt, model, context, **k: (
        asked.append(prompt[0]["text"]) or ('{"images": []}', {})))
    image_check.check(conversation, lambda image: "data:image/png;base64,eA==")
    assert asked == ["Inspect these images, in this order: image 1 (draft), image 2 (finished)."]


def test_a_job_records_how_long_comfyui_took(finishing):
    box, fake, conversation_id, _ = finishing
    box.run("run_workflow", json.dumps({"workflow": "krea-2/txt2img", "changes": {"positive": "x"}}))
    runs.tracker.poll_once()
    assert runs.tracker.waiting == {conversation_id: 1}  # still waiting
    fake.finish("prompt-1", filename="done.png", seconds=42.5)
    runs.tracker.poll_once()
    assert conversations.get_run(conversation_id, 1)["jobs"][0]["seconds"] == 42.5
    assert runs.tracker.waiting == {}
    assert comfy.execution_seconds([["execution_start", {"timestamp": 5000}], ["execution_cached", {}],
                                    ["execution_success", {"timestamp": 65_500}]]) == 60.5
    assert comfy.execution_seconds([["execution_start", {"timestamp": 5000}]]) is None


def _timed_run(n, workflow, seconds, kind=None):
    return {"workflow": workflow, "changes": {}, "count": len(seconds), "status": "done",
            **({"kind": kind} if kind else {}),
            "jobs": [{"prompt_id": f"p{n}-{i}", "status": "success", "seconds": s, "images": [],
                      "finished_at": f"2026-11-04T10:0{i}:00"} for i, s in enumerate(seconds)]}


def test_each_library_workflow_gets_its_time_per_image_and_the_queue_is_told():
    conversation = conversations.create("f", "m", 0.7)
    conversations.add_run(conversation["id"], _timed_run(1, "krea-2/txt2img", [50, 60, 70]), [])
    conversations.add_run(conversation["id"], _timed_run(2, "run:1", [58]), [])  # counts for krea-2/txt2img
    conversations.add_run(conversation["id"], _timed_run(3, "image:4", [300]), [])  # a shared image's: none
    conversations.add_run(conversation["id"], _timed_run(4, "krea-2/finish-image", [150, 170, 160]), [])

    measured = timing.load()
    assert measured["workflows"]["krea-2/txt2img"] == {"seconds": 59.0, "jobs": 4}
    assert "image:4" not in measured["workflows"] and measured["workflows"]["krea-2/finish-image"]["seconds"] == 160
    graph = Graph(workflows.find_by_name("krea-2/txt2img")["workflow"]) if workflows.find_by_name(
        "krea-2/txt2img") else Graph({})
    assert timing.speed("krea-2/txt2img", graph, measured) == "~60 s an image"
    assert timing.speed("krea-2/finish-image", graph, measured) == "~2.5 min an image"
    detailed = Graph({"1": {"class_type": "KSampler", "inputs": {}}, "2": {"class_type": "FaceDetailer", "inputs": {}},
                      "3": {"class_type": "ImageUpscaleWithModel", "inputs": {}}})
    assert timing.speed("other", detailed, measured) == "not timed yet; 2 sampling passes and a model upscale"

    assert timing.queue_text(conversation["id"], {}, 60) is None
    text = timing.queue_text(conversation["id"], {conversation["id"]: 3, "other": 9}, 60)
    assert "12 images of this app waiting or being made right now (3 from this conversation), about 12 minutes" in text

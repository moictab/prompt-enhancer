import json

import pytest

from app import chat_tools, comfy, conversations, runs, workflows

from .conftest import png_bytes

COMFY_GRAPH = {
    "3": {"class_type": "KSampler", "inputs": {"seed": 77, "steps": 20, "positive": ["6", 0], "negative": ["7", 0],
                                                "model": ["4", 0], "latent_image": ["5", 0]}},
    "4": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "a.safetensors"}},
    "5": {"class_type": "EmptyLatentImage", "inputs": {"width": 512, "height": 512, "batch_size": 1}},
    "6": {"class_type": "CLIPTextEncode", "inputs": {"text": "a cat", "clip": ["4", 1]}},
    "7": {"class_type": "CLIPTextEncode", "inputs": {"text": "blurry", "clip": ["4", 1]}},
    "8": {"class_type": "LoadImage", "inputs": {"image": "old.png"}},
    "9": {"class_type": "SaveImage", "inputs": {"filename_prefix": "x", "images": ["3", 0]}},
}


@pytest.fixture
def setup(fake_comfy):
    workflows.ensure_seeded()
    conversation = conversations.create("f", "m", 0.7, [])
    image = conversations.save_image(conversation["id"], png_bytes(prompt=COMFY_GRAPH), "image/png")
    conversations.append_message(conversation["id"], {"role": "user", "content": "mira", "images": [image]})
    return chat_tools.ToolBox(conversation["id"], fake_comfy), fake_comfy, conversation["id"]


def _run(box, **arguments):
    return box.run(arguments.pop("_name", "run_workflow"), json.dumps(arguments))


def test_run_a_library_workflow_with_changes(setup):
    box, fake, conversation_id = setup

    result, part = _run(box, workflow="krea-2/txt2img", count=2,
                        changes={"positive": "a witch at dawn", "steps": 10, "seed": "500",
                                 "loras": [{"name": "krea-2\\style-ink_v1.safetensors", "strength": 0.3}]})

    assert result["run"] == 1 and result["status"] == "queued" and result["images"] == 2
    assert result["seeds"] == [500, 501]
    assert part == {"type": "run", "run": 1, "workflow": "krea-2/txt2img", "count": 2}
    first, second = fake.submitted
    assert first["63"]["inputs"]["value"] == "a witch at dawn"
    assert first["53"]["inputs"]["steps"] == 10
    assert (first["53"]["inputs"]["seed"], second["53"]["inputs"]["seed"]) == (500, 501)
    assert first["29"]["inputs"]["filename_prefix"] == chat_tools.OUTPUT_PREFIX
    assert [n["inputs"]["lora_name"] for n in first.values() if n["class_type"] == "LoraLoader"] == [
        "krea-2\\style-ink_v1.safetensors"]
    # The library entry itself is untouched.
    assert workflows.find_by_name("krea-2/txt2img")["workflow"]["63"]["inputs"]["value"] == ""

    run = conversations.get_run(conversation_id, 1)
    assert run["status"] == "pending"
    assert [j["prompt_id"] for j in run["jobs"]] == ["prompt-1", "prompt-2"]
    assert conversations.run_workflows(conversation_id, 1) == [first, second]
    assert (conversation_id, 1) in runs.tracker._pending


def test_run_the_workflow_embedded_in_an_image_keeping_its_seed_and_feeding_an_image(setup):
    box, fake, conversation_id = setup
    image_id = conversations.get(conversation_id)["messages"][0]["images"][0]["id"]

    result, _ = _run(box, workflow="image:1", changes={"seed": "keep", "input_images": [{"node": "8", "image": 1}]})

    assert "error" not in result, result
    [sent] = fake.submitted
    assert sent["3"]["inputs"]["seed"] == 77
    # Uploaded into prompt-enhancer's own subfolder, named after the conversation and the image.
    assert fake.uploads == [f"{conversation_id}-{image_id}.png"]
    assert sent["8"]["inputs"]["image"] == f"prompt-enhancer/{conversation_id}-{image_id}.png"


def test_a_file_the_conversation_didnt_share_is_refused(setup):
    box, fake, _ = setup

    # The shared workflow's LoadImage still points at the user's own file in ComfyUI.
    result, part = _run(box, workflow="image:1")

    assert "node 8 (LoadImage) input image refers to 'old.png'" in result["error"]
    assert part["error"]
    assert fake.submitted == []


def test_the_isolation_holds_against_any_edit(setup):
    box, fake, conversation_id = setup
    other = "11111111-2222-3333-4444-555555555555"
    attempts = {
        "a node that reads ComfyUI's output folder": {
            "add": [{"id": "30", "class_type": "LoadImageOutput", "inputs": {"image": "ComfyUI_00001_.png [output]"}}]},
        "a node type nobody gave it": {
            "add": [{"id": "30", "class_type": "Some Custom Loader", "inputs": {"folder": "output"}}]},
        "a path": {"set": [{"node": "8", "input": "image", "value": "C:/Users/moict/Pictures/x.png"}]},
        "a URL": {"set": [{"node": "8", "input": "image", "value": "https://example.com/x.png"}]},
        "another conversation's upload": {
            "set": [{"node": "8", "input": "image", "value": f"prompt-enhancer/{other}-{'a' * 32}.png"}]},
        "an image of this conversation that doesn't exist": {
            "set": [{"node": "8", "input": "image", "value": f"prompt-enhancer/{conversation_id}-{'b' * 32}.png"}]},
    }
    for label, changes in attempts.items():
        result, _ = _run(box, workflow="image:1", changes=changes)
        assert "reaches outside" in result.get("error", ""), label
    assert fake.submitted == []


def test_prompts_with_links_or_file_names_in_their_text_are_fine(setup):
    box, fake, _ = setup

    result, _ = _run(box, workflow="image:1", changes={
        "remove": ["8"], "positive": "poster in the style of https://example.com, file photo.png on a desk"})

    assert "error" not in result, result


def test_saving_cant_widen_the_node_types(setup):
    box, _, _ = setup

    result, _ = _run(box, _name="save_workflow", workflow="image:1", name="mine/sneaky", description="x",
                     changes={"add": [{"id": "30", "class_type": "Some Custom Loader", "inputs": {}}]})

    assert "isn't a node type available here" in result["error"]
    assert workflows.find_by_name("mine/sneaky") is None


def test_rerun_an_earlier_run(setup):
    box, fake, _ = setup
    _run(box, workflow="image:1", changes={"positive": "a dog", "remove": ["8"]})

    _run(box, workflow="run:1")

    assert fake.submitted[1]["6"]["inputs"]["text"] == "a dog"


def test_errors_come_back_to_the_bot(setup):
    box, fake, conversation_id = setup

    assert "There is no workflow" in _run(box, workflow="nope")[0]["error"]
    assert "There is no image 5" in _run(box, workflow="image:5")[0]["error"]
    assert "negative" in _run(box, workflow="krea-2/txt2img", changes={"negative": "x"})[0]["error"]
    assert "doesn't exist" in _run(box, workflow="image:1", changes={"set": [
        {"node": "9", "input": "images", "value": ["404", 0]}]})[0]["error"]

    fake.reject = comfy.ComfyError(400, "Prompt outputs failed validation", {
        "141": {"class_type": "LoraLoader", "errors": [{"message": "Value not in list", "details": "lora_name"}]}})
    result, part = _run(box, workflow="krea-2/txt2img")
    assert result["error"] == "ComfyUI rejected it: Prompt outputs failed validation"
    assert result["node_errors"] == {"141": {"class_type": "LoraLoader", "errors": ["Value not in list: lora_name"]}}
    assert part["error"]

    fake.reject, fake.unreachable = None, True
    assert "PC may be off" in _run(box, workflow="krea-2/txt2img")[0]["error"]
    assert box.run("run_workflow", "not json")[0]["error"]
    assert box.run("fly", "{}")[0]["error"] == "There is no tool fly."
    assert (conversations.get(conversation_id).get("runs") or []) == []


def test_structural_edits(setup):
    box, fake, _ = setup

    result, _ = _run(box, workflow="image:1", changes={
        "add": [{"id": "20", "class_type": "ImageScaleBy", "inputs": {"image": ["3", 0], "scale_by": 2}}],
        "set": [{"node": "9", "input": "images", "value": ["20", 0]}],
        "remove": ["8"],
        "width": 768, "height": 1024,
    })

    assert "error" not in result, result
    [sent] = fake.submitted
    assert sent["20"]["class_type"] == "ImageScaleBy"
    assert sent["9"]["inputs"]["images"] == ["20", 0]
    assert "8" not in sent
    assert (sent["5"]["inputs"]["width"], sent["5"]["inputs"]["height"]) == (768, 1024)


def test_inspect_and_list_models(setup):
    box, fake, _ = setup

    inspected, part = _run(box, _name="inspect_workflow", workflow="image:1")
    assert inspected["prompt_inputs"] == {"positive": ["6.text"], "negative": ["7.text"]}
    assert inspected["image_inputs"] == ["8"]
    assert inspected["does"]["settings"]["seed"] == 77
    assert part == {"type": "tool", "tool": "inspect_workflow", "workflow": "image:1"}

    listed, part = _run(box, _name="list_comfy_models", folder="loras", search="KREA ink")
    assert listed == {"folder": "loras", "total": 1, "files": ["krea-2\\style-ink_v1.safetensors"]}
    assert part["found"] == 1
    assert "folder must be" in _run(box, _name="list_comfy_models", folder="bogus")[0]["error"]


def test_save_workflow_creates_and_replaces_only_when_asked(setup):
    box, _, _ = setup

    saved, part = _run(box, _name="save_workflow", workflow="image:1", name="mine/cat",
                       description="Cats.", changes={"steps": 30})
    assert saved == {"saved": "mine/cat", "replaced": False}
    assert workflows.find_by_name("mine/cat")["workflow"]["3"]["inputs"]["steps"] == 30

    again, _ = _run(box, _name="save_workflow", workflow="image:1", name="mine/cat", description="Cats.")
    assert "already a workflow" in again["error"]
    replaced, _ = _run(box, _name="save_workflow", workflow="image:1", name="mine/cat", description="Cats.",
                       replace=True)
    assert replaced["replaced"] is True
    assert workflows.find_by_name("mine/cat")["workflow"]["3"]["inputs"]["steps"] == 20


def test_cancel_run(setup):
    box, fake, conversation_id = setup
    _run(box, workflow="image:1", changes={"remove": ["8"]})

    result, _ = _run(box, _name="cancel_run", run=1)

    assert result == {"run": 1, "status": "cancelled"}
    assert fake.cancelled == ["prompt-1"]
    assert "already finished" in _run(box, _name="cancel_run", run=1)[0]["note"]
    assert "no run 9" in _run(box, _name="cancel_run", run=9)[0]["error"]

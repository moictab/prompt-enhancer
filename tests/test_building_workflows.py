"""The bot builds what an image needs: the base model it chose when writing the prompt, loaded in
whatever workflow it launches; workflows it writes from scratch; node types it looks up first."""
import json
from unittest.mock import patch

import pytest

from app import chat, chat_tools, conversations, families, workflows
from app.openrouter_client import StreamEnd

TURBO = "krea-2\\krea-2-turbo-official-comfy-int8-convrot_v1.0.safetensors"
GONZALOMO = "krea-2\\gonzalomo_v1.0.safetensors"
ANIMA = "anima\\anima_v-base-1.0.safetensors"
CHECKPOINT = "sdxl\\big-love_v3.safetensors"

WRITTEN = {  # a plain text-to-image workflow, as the bot would write it
    "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": CHECKPOINT}},
    "2": {"class_type": "CLIPTextEncode", "inputs": {"text": "a lighthouse in a storm", "clip": [1, 1]}},
    "3": {"class_type": "CLIPTextEncode", "inputs": {"text": "blurry", "clip": ["1", 1]}},
    "4": {"class_type": "EmptyLatentImage", "inputs": {"width": 1024, "height": 1024, "batch_size": 1}},
    "5": {"class_type": "KSampler", "inputs": {"model": ["1", 0], "positive": ["2", 0], "negative": ["3", 0],
                                               "latent_image": ["4", 0], "seed": 1, "steps": 25, "cfg": 6,
                                               "sampler_name": "euler", "scheduler": "normal", "denoise": 1}},
    "6": {"class_type": "VAEDecode", "inputs": {"samples": ["5", 0], "vae": ["1", 2]}},
    "7": {"class_type": "SaveImage", "inputs": {"images": ["6", 0], "filename_prefix": "x"}},
}


@pytest.fixture
def box(fake_comfy):
    workflows.ensure_seeded()
    fake_comfy.model_files.update(diffusion_models=[TURBO, GONZALOMO, ANIMA], checkpoints=[CHECKPOINT])
    conversation = conversations.create("f", "m", 0.7, [])
    return chat_tools.ToolBox(conversation["id"], fake_comfy)


def _run(box, _tool="run_workflow", **arguments):
    return box.run(_tool, json.dumps(arguments))


def _unet(workflow: dict) -> str:
    return next(n["inputs"]["unet_name"] for n in workflow.values() if n["class_type"] == "UNETLoader")


def test_the_prompt_runs_on_the_model_its_block_names(api_client, auth_headers, tmp_path, fake_comfy):
    workflows.ensure_seeded()
    fake_comfy.model_files["diffusion_models"] = [TURBO, GONZALOMO]
    family = families.create_family("Krea", "rules", False, path=str(tmp_path / "families.json"))
    conversation_id = api_client.post("/api/conversations", auth=auth_headers,
                                      json={"family_id": family["id"], "llm_model": "vendor/model"}).json()["id"]
    run_call = {"id": "c1", "name": "run_workflow",
                "arguments": json.dumps({"workflow": "krea-2/txt2img", "changes": {"positive": "a witch at dawn"}})}

    def script(api_key, model, messages, temperature, context=None, tools=None, tool_choice=None):
        yield f'Gonzalomo, por la piel:\n<prompt model="{GONZALOMO}">\nPOSITIVE: a witch at dawn\n</prompt>'
        yield StreamEnd(cost=0.01, tool_calls=[run_call])

    with patch("app.conversation_turn.stream_chat", script):
        response = api_client.post(f"/api/conversations/{conversation_id}/messages", data={"content": "una bruja"},
                                   auth=auth_headers)
    events = [json.loads(line) for line in response.text.splitlines() if line.strip()]
    assert events[-1]["type"] == "done", events[-1]

    # The library workflow loads the turbo model; the block said gonzalomo, and that is what ran.
    assert _unet(fake_comfy.submitted[0]) == GONZALOMO
    stored = conversations.get(conversation_id)
    assert stored["runs"][0]["base_models"] == [GONZALOMO]
    result = json.loads(stored["messages"][-1]["steps"][0]["results"][0])
    assert result["model"] == GONZALOMO
    assert f"model {GONZALOMO}, as your <prompt> block says" in result["applied"]


def test_an_earlier_block_names_the_model_of_a_prompt_run_again(box, fake_comfy):
    conversations.append_message(box.conversation_id, {
        "role": "assistant", "content": "x",
        "parts": [{"type": "prompt", "positive_prompt": "a fox in the snow", "negative_prompt": "", "model": GONZALOMO}]})

    _run(box, workflow="krea-2/txt2img", changes={"positive": "A fox in the  snow"})

    assert _unet(fake_comfy.submitted[0]) == GONZALOMO


def test_changes_model_loads_another_model_where_the_workflow_can(box, fake_comfy):
    result, _ = _run(box, workflow="krea-2/txt2img", changes={"model": "gonzalomo_v1.0", "positive": "a cat"})
    assert "error" not in result and _unet(fake_comfy.submitted[0]) == GONZALOMO

    # A checkpoint can't go where the workflow loads a diffusion model on its own.
    result, _ = _run(box, workflow="krea-2/txt2img", changes={"model": CHECKPOINT})
    assert result["error"].startswith(f"{CHECKPOINT} is a checkpoint, but this workflow loads a diffusion model")
    # Nor a diffusion model of another family: the text encoders and VAE are krea-2's.
    result, _ = _run(box, workflow="krea-2/txt2img", changes={"model": ANIMA})
    assert "text encoders and VAE are for krea-2" in result["error"]
    result, _ = _run(box, workflow="krea-2/txt2img", changes={"model": "nope.safetensors"})
    assert "isn't an installed checkpoint or diffusion model" in result["error"]
    assert len(fake_comfy.submitted) == 1


def test_a_block_naming_a_model_the_workflow_cant_load_stops_the_launch(box, fake_comfy):
    box.reply_text = f'<prompt model="{ANIMA}">\nPOSITIVE: a cat\n</prompt>'

    result, _ = _run(box, workflow="krea-2/txt2img", changes={"positive": "a cat"})

    assert result["error"].startswith(f"Your <prompt> block says {ANIMA}, which can't run here:")
    assert fake_comfy.submitted == []
    # Named explicitly, another model goes.
    result, _ = _run(box, workflow="krea-2/txt2img", changes={"positive": "a cat", "model": TURBO})
    assert "error" not in result


def test_a_workflow_written_from_scratch_runs_and_can_be_run_again(box, fake_comfy):
    result, part = _run(box, graph=WRITTEN, changes={"aspect": "16:9"})

    assert "error" not in result, result
    assert part == {"type": "run", "run": 1, "workflow": "custom", "count": 1}
    sent = fake_comfy.submitted[0]
    assert sent["2"]["inputs"]["clip"] == ["1", 1]  # a link with a numeric id, normalized
    assert sent["7"]["inputs"]["filename_prefix"].startswith("prompt-enhancer/")
    assert result["model"] == CHECKPOINT and result["size"] == "1360x768 (16:9 landscape)"

    again, _ = _run(box, workflow="run:1", changes={"positive": "a lighthouse at dawn"})
    assert "error" not in again and fake_comfy.submitted[1]["2"]["inputs"]["text"] == "a lighthouse at dawn"

    saved, _ = _run(box, "save_workflow", graph=WRITTEN, name="sdxl/plain", description="Plain SDXL text to image.")
    assert saved == {"saved": "sdxl/plain", "replaced": False}


def test_a_written_workflow_is_checked_like_any_other(box, fake_comfy):
    unsaved = {k: v for k, v in WRITTEN.items() if k != "7"}
    assert _run(box, graph=unsaved)[0]["error"] == "Your workflow saves nothing: end it in a SaveImage node."
    assert "both" in _run(box, graph=WRITTEN, workflow="krea-2/txt2img")[0]["error"]
    assert "Links to nodes that don't exist: 5.model" in _run(
        box, graph={**WRITTEN, "5": {**WRITTEN["5"], "inputs": {**WRITTEN["5"]["inputs"], "model": ["99", 0]}}})[0]["error"]
    sneaky = {**WRITTEN, "8": {"class_type": "LoadImageOutput", "inputs": {"image": "other-project/x.png"}}}
    assert "is never allowed" in _run(box, graph=sneaky)[0]["error"]
    assert fake_comfy.submitted == []


def test_describe_nodes_shows_the_available_types_and_never_the_input_folders_files(box, fake_comfy):
    fake_comfy.node_definitions = {
        "KSampler": {"input": {"required": {"seed": ["INT", {"default": 0, "min": 0, "max": 2 ** 64}],
                                            "sampler_name": [["euler", "dpmpp_2m"], {}],
                                            "model": ["MODEL"]}},
                     "output": ["LATENT"], "output_name": ["LATENT"], "description": "Denoises a latent."},
        "LoadImage": {"input": {"required": {"image": [["other-project/secret.png", "a.jpg"],
                                                       {"image_upload": True, "default": "other-project/secret.png"}]}},
                      "output": ["IMAGE", "MASK"], "output_name": ["IMAGE", "MASK"]},
    }

    listed, _ = _run(box, "describe_nodes")
    assert {"KSampler", "UNETLoader", "SaveImage"} <= set(listed["available"])
    assert "LoadImageOutput" not in listed["available"]

    described, part = _run(box, "describe_nodes", types=["KSampler", "LoadImage", "LoadImageOutput"])
    sampler = described["nodes"]["KSampler"]
    assert sampler["inputs"]["required"]["sampler_name"] == {"type": "COMBO", "choices": ["euler", "dpmpp_2m"]}
    assert sampler["inputs"]["required"]["seed"] == {"type": "INT", "default": 0, "min": 0, "max": 2 ** 64}
    assert sampler["outputs"] == [{"index": 0, "type": "LATENT", "name": "LATENT"}]
    loader = json.dumps(described["nodes"]["LoadImage"])
    assert "secret" not in loader and "this conversation's" in loader
    assert "not a node type available here" in described["nodes"]["LoadImageOutput"]["error"].lower()
    assert part == {"type": "tool", "tool": "describe_nodes", "types": ["KSampler", "LoadImage", "LoadImageOutput"]}


def test_the_bot_is_told_to_decide_the_model_when_writing_and_to_build_freely():
    assert "Decide the base model when you write a prompt, not when you launch it" in chat.COMFY_MECHANICS
    assert "Workflows are means, not limits" in chat.COMFY_MECHANICS
    names = {t["function"]["name"] for t in chat_tools.TOOLS}
    assert {"describe_nodes", "run_workflow"} <= names
    run = next(t for t in chat_tools.TOOLS if t["function"]["name"] == "run_workflow")["function"]["parameters"]
    assert "graph" in run["properties"] and "model" in run["properties"]["changes"]["properties"]

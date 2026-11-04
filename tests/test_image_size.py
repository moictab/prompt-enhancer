"""The bot picks an image's proportions (aspect) and the workflow keeps its own pixel count."""
import json
from pathlib import Path

import pytest

from app import chat, chat_tools, conversation_turn, conversations, workflows
from app.comfy_graph import Graph, GraphError, normalize_aspect

SEEDS = Path(__file__).parent.parent / "app" / "seed_workflows"

FIXED = {
    "3": {"class_type": "KSampler", "inputs": {"seed": 1, "model": ["4", 0], "latent_image": ["5", 0]}},
    "4": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "a.safetensors"}},
    "5": {"class_type": "EmptyLatentImage", "inputs": {"width": 1024, "height": 1024, "batch_size": 1}},
    "9": {"class_type": "SaveImage", "inputs": {"filename_prefix": "x", "images": ["3", 0]}},
}


def _seed(name):
    return Graph(json.loads((SEEDS / name).read_text(encoding="utf-8")))


def test_aspect_names():
    assert normalize_aspect("2:3 (Portrait Photo)") == "2:3"
    assert normalize_aspect(" 16/9 ") == "16:9"
    assert normalize_aspect("5:4") is None
    assert normalize_aspect(None) is None


def test_a_resolution_selector_takes_its_option_and_keeps_its_megapixels():
    graph = _seed("krea-2_txt2img.json")
    assert graph.shape() == "2:3 portrait, 1 MP"

    graph.set_aspect("16:9")

    assert graph.nodes["49"]["inputs"]["aspect_ratio"] == "16:9 (Widescreen)"
    assert graph.nodes["49"]["inputs"]["megapixels"] == 1.0
    assert graph.shape() == "16:9 landscape, 1 MP"


def test_a_fixed_size_is_recomputed_for_the_same_pixel_count():
    graph = Graph(json.loads(json.dumps(FIXED)))
    assert graph.shape() == "1024x1024 (1:1 square)"

    graph.set_aspect("2:3")

    inputs = graph.nodes["5"]["inputs"]
    assert (inputs["width"], inputs["height"]) == (832, 1248)
    assert graph.shape() == "832x1248 (2:3 portrait)"


def test_a_fixed_rescale_further_on_takes_the_new_proportions():
    """The hires pass rescales to 1560x1560: left alone, a portrait would come out squashed."""
    graph = _seed("krea-2_txt2img-detailer-hires.json")

    graph.set_aspect("2:3")

    scale = graph.nodes["146:108"]["inputs"]
    assert abs(scale["width"] / scale["height"] - 2 / 3) < 0.01
    assert abs(scale["width"] * scale["height"] / 1560 ** 2 - 1) < 0.02
    assert scale["width"] % 8 == 0 and scale["height"] % 8 == 0

    exact = _seed("krea-2_txt2img-detailer-hires.json")
    exact.set_size(width=1216, height=832)
    scale = exact.nodes["146:108"]["inputs"]
    assert scale["width"] > scale["height"]


def test_an_unknown_aspect_or_a_workflow_without_a_size_is_an_error():
    with pytest.raises(GraphError, match="unknown aspect"):
        _seed("krea-2_txt2img.json").set_aspect("5:4")
    with pytest.raises(GraphError, match="input image"):
        _seed("krea-2_face-detail-image.json").set_aspect("2:3")


@pytest.fixture
def toolbox(fake_comfy):
    workflows.ensure_seeded()
    conversation = conversations.create("f", "m", 0.7, [])
    conversations.append_message(conversation["id"], {"role": "user", "content": "hola", "images": []})
    return chat_tools.ToolBox(conversation["id"], fake_comfy), fake_comfy, conversation["id"]


def test_the_bot_sets_the_aspect_and_the_run_remembers_the_size(toolbox):
    box, fake, conversation_id = toolbox

    result, _ = box.run("run_workflow", json.dumps({"workflow": "krea-2/txt2img", "changes": {"aspect": "16:9"}}))

    assert fake.submitted[0]["49"]["inputs"]["aspect_ratio"] == "16:9 (Widescreen)"
    assert result["size"] == "16:9 landscape, 1 MP"
    assert "aspect 16:9" in result["applied"]
    run = conversations.get_run(conversation_id, 1)
    assert run["size"] == "16:9 landscape, 1 MP"
    assert chat.results_text({**run, "jobs": []}, {}, False).startswith(
        "[Run 1 (krea-2/txt2img, 16:9 landscape, 1 MP) produced no images.")

    # The older name still works, with the node's label or without.
    box.run("run_workflow", json.dumps({"workflow": "krea-2/txt2img", "changes": {"aspect_ratio": "3:2 (Photo)"}}))
    assert fake.submitted[1]["49"]["inputs"]["aspect_ratio"] == "3:2 (Photo)"


def test_the_tool_offers_the_aspects():
    run_workflow = next(t for t in chat_tools.TOOLS if t["function"]["name"] == "run_workflow")
    changes = run_workflow["function"]["parameters"]["properties"]["changes"]["properties"]
    assert changes["aspect"]["enum"] == ["1:1", "2:3", "3:2", "3:4", "4:3", "9:16", "16:9", "21:9"]
    assert "aspect_ratio" not in changes


def test_the_library_shows_each_workflows_default_size():
    workflows.ensure_seeded()
    entry = conversation_turn._library_entry(workflows.find_by_name("krea-2/txt2img"))
    assert entry["size"] == "2:3 portrait, 1 MP"

    family = {"name": "Krea 2", "instructions": "rules", "has_negative_prompt": False}
    system = chat.build_system_prompt(family, workflows=[{**entry, "family": "Krea 2"}],
                                      families=[])

    assert ("- krea-2/txt2img [Krea 2] on krea-2\\krea-2-turbo-official-comfy-int8-convrot_v1.0.safetensors "
            "(2:3 portrait, 1 MP by default)") in system
    assert "Choose the image's proportions yourself" in system

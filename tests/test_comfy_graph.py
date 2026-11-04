import json
from pathlib import Path

import pytest

from app.comfy_graph import Graph, GraphError, is_api_workflow, is_editor_workflow

SEEDS = Path(__file__).parent.parent / "app" / "seed_workflows"


def _seed(name):
    return Graph(json.loads((SEEDS / name).read_text(encoding="utf-8")))


def test_seed_workflows_are_api_format():
    for path in SEEDS.glob("*.json"):
        if path.name != "index.json":
            assert is_api_workflow(json.loads(path.read_text(encoding="utf-8"))), path.name
    assert not is_api_workflow({"nodes": [], "links": []})
    assert is_editor_workflow({"nodes": [], "links": []})


def test_prompt_slots_follow_the_main_sampler():
    krea = _seed("krea-2_txt2img.json")
    illustrious = _seed("illustrious_txt2img.json")

    assert krea.prompt_slots() == {"positive": [("63", "value")], "negative": [], "other": []}
    slots = illustrious.prompt_slots()
    assert slots["positive"] == [("150", "value")]
    assert slots["negative"] == [("173", "value")]


def test_set_prompt_writes_both_texts_and_refuses_a_negative_where_there_is_none():
    graph = _seed("illustrious_txt2img.json")
    graph.set_prompt("a witch", "blurry")
    assert graph.nodes["150"]["inputs"]["value"] == "a witch"
    assert graph.nodes["173"]["inputs"]["value"] == "blurry"

    krea = _seed("krea-2_txt2img.json")
    with pytest.raises(GraphError):
        krea.set_prompt("a witch", "blurry")
    krea.set_prompt("a witch", "")  # an empty negative is fine
    assert krea.nodes["63"]["inputs"]["value"] == "a witch"


def test_set_loras_replaces_the_stack_and_rewires_its_users():
    graph = _seed("krea-2_txt2img.json")
    assert len(graph.lora_chain()) == 6

    created = graph.set_loras([{"name": "a.safetensors", "strength": 0.4}, {"name": "b.safetensors", "strength": 1}])

    chain = graph.lora_chain()
    assert [c["name"] for c in chain] == ["a.safetensors", "b.safetensors"]
    assert [c["node"] for c in chain] == created
    assert graph.nodes[created[0]]["inputs"]["model"] == ["55", 0]  # the UNET loader
    assert graph.nodes[created[0]]["inputs"]["clip"] == ["56", 0]
    assert graph.nodes["53"]["inputs"]["model"] == [created[1], 0]  # the sampler uses the new stack
    assert graph.nodes["51"]["inputs"]["clip"] == [created[1], 1]
    assert graph.broken_links() == []

    graph.set_loras([])
    assert graph.lora_chain() == []
    assert graph.nodes["53"]["inputs"]["model"] == ["55", 0]


def _weighted(graph):
    """krea-2/txt2img as in the library since Krea2PromptWeight: it encodes the prompt and patches the model."""
    encoder = graph.nodes.pop("51")["inputs"]
    graph.nodes["149"] = {"class_type": "Krea2PromptWeight", "inputs": {
        "clip": encoder["clip"], "model": ["147", 0], "text": encoder["text"], "strength": 1.0}}
    graph.nodes["53"]["inputs"].update(model=["149", 0], positive=["149", 1])
    graph.nodes["58"]["inputs"]["conditioning"] = ["149", 1]
    return graph


def test_a_prompt_weighting_node_keeps_the_prompt_and_the_loras_before_it():
    graph = _weighted(_seed("krea-2_txt2img.json"))
    assert graph.prompt_slots()["positive"] == [("63", "value")]

    created = graph.set_loras([{"name": "a.safetensors", "strength": 0.4}])
    assert graph.nodes["149"]["inputs"]["model"] == [created[0], 0]
    assert graph.nodes["149"]["inputs"]["clip"] == [created[0], 1]
    assert graph.nodes["53"]["inputs"]["model"] == ["149", 0]

    # No LoRAs left, then some again: they go between the loaders and the weighting, not after it.
    graph.set_loras([])
    assert graph.nodes["149"]["inputs"]["model"] == ["55", 0]
    created = graph.set_loras([{"name": "b.safetensors", "strength": 1}])
    assert graph.nodes[created[0]]["inputs"]["model"] == ["55", 0]
    assert graph.nodes[created[0]]["inputs"]["clip"] == ["56", 0]
    assert graph.nodes["149"]["inputs"]["model"] == [created[0], 0]
    assert graph.nodes["53"]["inputs"]["model"] == ["149", 0]
    assert graph.broken_links() == []


def test_set_size_replaces_the_resolution_selector_and_seed_reaches_every_sampler():
    graph = _seed("krea-2_txt2img-detailer-upscale.json")

    graph.set_size(width=1024, height=1536)
    graph.set_seed(99)

    assert graph.nodes["52"]["inputs"]["width"] == 1024
    assert "49" not in graph.nodes  # nothing else used the ResolutionSelector
    seeds = [n["inputs"]["seed"] for n in graph.nodes.values() if "seed" in n["inputs"]]
    assert len(seeds) > 1 and set(seeds) == {99}
    assert graph.seed() == 99


def test_set_sampler_and_generation_info():
    graph = _seed("krea-2_txt2img.json")
    graph.set_prompt("a witch")
    graph.set_sampler(steps=12, cfg=1.5)

    info = graph.generation_info()

    assert info["positive"] == "a witch"
    assert "negative" not in info
    assert info["settings"]["steps"] == 12
    assert info["settings"]["size"] == "2:3 (Portrait Photo), 1.0 MP"
    assert "krea-2\\krea-2-turbo-official-comfy-int8-convrot_v1.0.safetensors" in info["models"]
    assert len(info["loras"]) == 6
    with pytest.raises(GraphError):
        graph.set_sampler(steps=1, bogus=2)


def test_bypass_wires_consumers_to_the_matching_input():
    graph = _seed("krea-2_txt2img.json")

    graph.bypass("147")  # the last LoRA, feeding the sampler and the text encoder

    assert graph.nodes["53"]["inputs"]["model"] == ["146", 0]
    assert graph.nodes["51"]["inputs"]["clip"] == ["146", 1]
    assert "147" not in graph.nodes


def test_add_and_set_inputs_validate_links_and_ids():
    graph = _seed("krea-2_txt2img.json")

    node_id = graph.add_node("ImageScaleBy", {"image": ["138", 0], "upscale_method": "lanczos", "scale_by": 1.5},
                             node_id="200")
    graph.set_input("29", "images", [node_id, 0])

    assert graph.nodes["29"]["inputs"]["images"] == ["200", 0]
    with pytest.raises(GraphError):
        graph.add_node("X", node_id="200")
    with pytest.raises(GraphError):
        graph.set_input("29", "images", ["nope", 0])
    graph.nodes["29"]["inputs"]["images"] = ["gone", 0]
    assert graph.broken_links() == ["29.images -> gone"]


def test_rgthree_style_lora_stacks_are_read():
    graph = Graph({"1": {"class_type": "Power Lora Loader (rgthree)", "inputs": {
        "lora_1": {"on": True, "lora": "a.safetensors", "strength": 0.6},
        "lora_2": {"on": False, "lora": "b.safetensors", "strength": 1},
        "PowerLoraLoaderHeaderWidget": {"type": "PowerLoraLoaderHeaderWidget"},
    }}})

    assert graph.loras() == [{"node": "1", "name": "a.safetensors", "strength": 0.6}]


def test_summary_is_compact_and_ordered():
    graph = Graph({
        "10": {"class_type": "B", "inputs": {"x": ["9", 0]}},
        "9": {"class_type": "A", "inputs": {"text": "y" * 200}, "_meta": {"title": "Prompt"}},
    })

    summary = graph.summary()

    assert [n["id"] for n in summary] == ["9", "10"]
    assert summary[0]["title"] == "Prompt"
    assert summary[0]["values"]["text"].endswith("(200 characters)")
    assert summary[1]["links"] == {"x": "9.0"}

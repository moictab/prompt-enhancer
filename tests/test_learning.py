"""The bot learns from its runs: it reviews every image, the review is kept with the image's
recipe, the user's dislikes carry their reason, and LoRAs can be calibrated weight by weight.
Plus switching family by name."""
import json
from unittest.mock import patch

from app import catalog, chat, chat_tools, conversation_turn, conversations, families, runs, workflows

from .conftest import png_bytes
from .test_api_conversar_comfy import Script

INK = "krea-2\\style-ink_v1.safetensors"


def test_parse_reply_reads_reviews_and_keeps_them_out_of_the_text():
    reply = chat.parse_reply(
        'La 2 falla. <review image="7" verdict="ok">clean</review>\n<review image="8" verdict="flawed">fused hands; '
        "ink LoRA at 0.8</review>\n<review image=9>ok, fine</review>\nBajo el peso.",
        has_negative_prompt=False,
    )

    assert reply.reviews == [(7, "ok", "clean"), (8, "flawed", "fused hands; ink LoRA at 0.8"), (9, "ok", "ok, fine")]
    assert reply.parts == [{"type": "text", "text": "La 2 falla.\n\nBajo el peso."}]


def test_the_bot_sees_every_image_that_arrived_since_it_last_spoke():
    images = [{"id": f"i{n}", "content_type": "image/png"} for n in range(10)]
    conversation = {"id": "c", "runs": [], "messages": [
        {"role": "user", "content": "x", "images": images[:2]},
        {"role": "assistant", "content": "Lanzo."},
        {"role": "results", "run": 1, "content": "", "images": images[2:6]},
        {"role": "results", "run": 2, "content": "", "images": images[6:10]},
    ]}

    messages = chat.build_llm_messages(conversation, load_image=lambda image: f"data:{image['id']}")

    attached = [p["image_url"]["url"] for m in messages if isinstance(m["content"], list)
                for p in m["content"] if p["type"] == "image_url"]
    assert attached == [f"data:i{n}" for n in range(2, 10)]  # all 8 new ones, not just the last 4


def test_calibration_results_are_described_by_weight():
    run = {"n": 3, "workflow": "krea-2/txt2img", "kind": "calibration", "lora": INK, "jobs": [
        {"status": "success", "seed": 5, "images": ["a"], "calibration_weight": 0},
        {"status": "success", "seed": 5, "images": ["b"], "calibration_weight": 0.5}]}

    text = chat.results_text(run, {"a": 4, "b": 5}, automatic=False)

    assert text == f"[Run 3 (calibration of {INK} on krea-2/txt2img, seed 5) finished: image 4 (weight 0); image 5 (weight 0.5).]"


def _setup(fake_comfy):
    workflows.ensure_seeded()
    krea = families.create_family("Krea 2", "Krea rules.", False)
    families.create_family("Illustrious", "Tag rules.", True)
    conversation = conversations.create(krea["id"], "m", 0.7, [])
    conversations.append_message(conversation["id"], {"role": "user", "content": "hazla", "images": []})
    return chat_tools.ToolBox(conversation["id"], fake_comfy), conversation["id"]


def test_use_family_switches_by_name_and_hands_over_the_rules(fake_comfy):
    box, conversation_id = _setup(fake_comfy)

    result, part = box.run("use_family", json.dumps({"family": "illustrious"}))

    assert result == {"family": "Illustrious", "uses_negative_prompt": True, "family_rules": "Tag rules."}
    assert part == {"type": "family", "family": "Illustrious", "previous": "Krea 2", "reason": "asked"}
    assert families.get_family(conversations.get(conversation_id)["family_id"])["name"] == "Illustrious"
    assert box.run("use_family", json.dumps({"family": "Illustrious"}))[0]["note"].startswith("Already")
    assert "There is no family" in box.run("use_family", json.dumps({"family": "Pony"}))[0]["error"]


def test_the_system_prompt_names_the_current_family_and_the_others():
    result = chat.build_system_prompt({"name": "Krea 2", "instructions": "rules"}, [], [], needs_title=False,
                                      workflows=[], families=["Krea 2", "Illustrious", "Anima"])

    assert "## Model Family Rules: Krea 2\n" in result
    assert "## Model Families\nThe current one is Krea 2; the others: Illustrious, Anima." in result


def test_calibrate_lora_runs_one_image_per_weight_same_seed_at_idle_priority(fake_comfy):
    box, conversation_id = _setup(fake_comfy)

    result, part = box.run("calibrate_lora", json.dumps({
        "lora": "krea-2/style-ink_v1", "prompt": "a witch, ink linework", "weights": [0, 0.5, 1]}))

    assert result["lora"] == INK and result["workflow"] == "krea-2/txt2img" and result["weights"] == [0, 0.5, 1]
    assert part["calibration"] == INK and part["count"] == 3
    assert fake_comfy.priorities == ["idle"] * 3
    seeds = {next(n["inputs"]["seed"] for n in sent.values() if n["class_type"] == "KSampler")
             for sent in fake_comfy.submitted}
    assert seeds == {result["seed"]}
    strengths = [[n["inputs"]["strength_model"] for n in sent.values()
                  if n["class_type"] == "LoraLoader" and n["inputs"]["lora_name"] == INK] for sent in fake_comfy.submitted]
    assert strengths == [[0.0], [0.5], [1.0]]
    # The workflow's own LoRAs stay (the usual stack) unless asked otherwise.
    assert sum(1 for n in fake_comfy.submitted[0].values() if n["class_type"] == "LoraLoader") == 7
    run = conversations.get_run(conversation_id, 1)
    assert run["kind"] == "calibration" and run["lora"] == INK
    assert run["timeout_seconds"] == chat_tools.CALIBRATION_TIMEOUT_SECONDS
    assert [j["calibration_weight"] for j in run["jobs"]] == [0, 0.5, 1]
    assert fake_comfy.submitted[1]["29"]["inputs"]["filename_prefix"].endswith(f"seed-{result['seed']}_w-0.5")

    alone, _ = box.run("calibrate_lora", json.dumps({"lora": INK, "prompt": "x", "keep_other_loras": False}))
    assert alone["other_loras"] == []
    assert "between -2 and 3" in box.run("calibrate_lora", json.dumps({"lora": INK, "prompt": "x", "weights": [9, 1]}))[0]["error"]


def _finish_run_with_reviews(api_client, auth_headers, fake_comfy, conversation_id, reviews_text):
    with patch("app.conversation_turn.stream_chat", Script(("", [{"id": "c1", "name": "run_workflow", "arguments": json.dumps(
            {"workflow": "krea-2/txt2img", "count": 2, "changes": {"loras": [{"name": INK, "strength": 0.8}]}})}]),
            ("Va.", []))):
        api_client.post(f"/api/conversations/{conversation_id}/messages", auth=auth_headers, data={"content": "dos"})
    run_n = len(conversations.get(conversation_id)["runs"])
    with patch("app.conversation_turn.stream_chat", Script((reviews_text, []))):
        for job in conversations.get_run(conversation_id, run_n)["jobs"]:
            fake_comfy.finish(job["prompt_id"], f"out-{job['prompt_id']}")
        runs.tracker.poll_once()
        runs.wait_until_idle()


def test_reviews_are_kept_with_the_recipe_and_counted_per_file(api_client, auth_headers, fake_comfy):
    _, conversation_id = _setup(fake_comfy)

    _finish_run_with_reviews(api_client, auth_headers, fake_comfy, conversation_id,
                             'Una bien, otra no. <review image="1" verdict="ok">clean</review>'
                             '<review image="2" verdict="flawed">fused hands; ink too strong</review>')

    results = conversations.get(conversation_id)["messages"][-2]
    assert [img["review"]["verdict"] for img in results["images"]] == ["ok", "flawed"]
    record = catalog.stats()[INK]
    assert record["images"] == 2 and record["flawed"] == 1 and record["flawed_weights"] == [0.8, 0.8]
    assert record["latest_flaws"] == ["at 0.8: fused hands; ink too strong"]
    assert "style-ink_v1 (2 images, 1 flawed at 0.8)" in (catalog.refresh(fake_comfy) and catalog.prompt_section())

    # The user's word beats the bot's: liked, it isn't counted as flawed.
    api_client.put(f"/api/conversations/{conversation_id}/images/{results['images'][1]['id']}/reaction",
                   auth=auth_headers, json={"rating": "up"})
    assert "flawed" not in catalog.stats()[INK]


def test_a_thumbs_down_needs_its_reason(api_client, auth_headers, fake_comfy):
    _, conversation_id = _setup(fake_comfy)
    image = conversations.save_image(conversation_id, png_bytes(), "image/png")
    conversations.append_message(conversation_id, {"role": "results", "run": 1, "content": "", "images": [image]})
    put = f"/api/conversations/{conversation_id}/images/{image['id']}/reaction"

    assert api_client.put(put, auth=auth_headers, json={"rating": "down"}).status_code == 400
    assert api_client.put(put, auth=auth_headers, json={"rating": "down", "comment": "  "}).status_code == 400
    assert api_client.put(put, auth=auth_headers, json={"rating": "down", "comment": "body horror"}).status_code == 200


def test_calibration_verdicts_become_the_loras_calibration(api_client, auth_headers, fake_comfy):
    _, conversation_id = _setup(fake_comfy)
    box = chat_tools.ToolBox(conversation_id, fake_comfy)
    box.run("calibrate_lora", json.dumps({"lora": INK, "prompt": "x", "weights": [0, 0.5, 1]}))
    with patch("app.conversation_turn.stream_chat", Script((
            '<review image="1" verdict="ok">no ink</review><review image="2" verdict="ok">inky</review>'
            '<review image="3" verdict="flawed">burnt, broken faces</review>', []))):
        for job in conversations.get_run(conversation_id, 1)["jobs"]:
            fake_comfy.finish(job["prompt_id"], f"out-{job['prompt_id']}")
        runs.tracker.poll_once()
        runs.wait_until_idle()

    record = catalog.stats()[INK]
    assert record["calibration"] == {"0": "ok", "0.5": "ok", "1": "flawed: burnt, broken faces"}
    assert catalog._record_text(record).endswith("calibrated: fine up to 0.5, flawed from 1")


def test_turn_offers_the_family_list(api_client, auth_headers, fake_comfy):
    _, conversation_id = _setup(fake_comfy)
    with patch("app.conversation_turn.stream_chat", Script(("Hola.", []))) as script:
        api_client.post(f"/api/conversations/{conversation_id}/reply", auth=auth_headers)

    system = script.calls[0]["messages"][0]["content"]
    assert "The current one is Krea 2; the others: Illustrious." in system
    assert any(t["function"]["name"] == "use_family" for t in script.calls[0]["tools"])
    assert conversation_turn.chat_tools.TOOLS is script.calls[0]["tools"]

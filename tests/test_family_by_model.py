"""The conversation's family follows the base model the bot runs: models live in a folder
named after their family, and prompts follow that family's rules."""
import json
from unittest.mock import patch

from app import chat, chat_tools, conversations, families, history, workflows
from app.comfy_graph import Graph

from .test_api_conversar_comfy import Script


def _families():
    krea = families.create_family("Krea 2", "Write natural sentences.", False)
    ages = families.create_family("Krea 2 - Ages", "Krea for the ages project.", False)
    illustrious = families.create_family("Illustrious", "Write danbooru tags.", True)
    return krea, ages, illustrious


def test_a_model_folder_names_its_family():
    krea, ages, illustrious = _families()

    assert families.family_for_folder("krea-2")["id"] == krea["id"]
    assert families.family_for_folder("Illustrious")["id"] == illustrious["id"]
    assert families.family_for_folder("moss") is None
    assert families.covers_folder(krea, "krea-2") and families.covers_folder(ages, "krea-2")
    assert not families.covers_folder(illustrious, "krea-2")
    assert not families.covers_folder(krea, "krea")


def test_the_base_model_folder_of_a_workflow():
    def graph(*names, key="ckpt_name"):
        return Graph({str(i): {"class_type": "Loader", "inputs": {key: n}} for i, n in enumerate(names)})

    assert graph("illustrious\\wai.safetensors").base_model_folder() == "illustrious"
    assert graph("krea-2/turbo.safetensors", key="unet_name").base_model_folder() == "krea-2"
    assert graph("loose.safetensors").base_model_folder() is None
    assert graph("krea-2\\a", "illustrious\\b").base_model_folder() is None  # ambiguous
    # LoRAs, VAEs and text encoders don't decide it.
    assert Graph({"1": {"class_type": "LoraLoader", "inputs": {"lora_name": "anima\\x"}}}).base_model_folder() is None


def _box(fake_comfy, family):
    workflows.ensure_seeded()
    conversation = conversations.create(family["id"], "m", 0.7, [])
    conversations.append_message(conversation["id"], {"role": "user", "content": "hazla", "images": []})
    return chat_tools.ToolBox(conversation["id"], fake_comfy), conversation["id"]


def test_running_another_familys_model_switches_the_conversation_first(fake_comfy):
    krea, _, illustrious = _families()
    box, conversation_id = _box(fake_comfy, krea)

    result, part = box.run("run_workflow", json.dumps({"workflow": "illustrious/txt2img",
                                                        "changes": {"positive": "a witch at dawn"}}))

    assert "Not run yet" in result["error"]
    assert result["family"] == "Illustrious" and result["uses_negative_prompt"] is True
    assert result["family_rules"] == "Write danbooru tags."
    assert part == {"type": "family", "family": "Illustrious", "previous": "Krea 2", "reason": "model"}
    assert conversations.get(conversation_id)["family_id"] == illustrious["id"]
    assert fake_comfy.submitted == []

    result, _ = box.run("run_workflow", json.dumps({"workflow": "illustrious/txt2img",
                                                     "changes": {"positive": "1girl, witch, dawn"}}))
    assert result["run"] == 1 and len(fake_comfy.submitted) == 1


def test_a_variant_family_of_the_models_folder_is_kept(fake_comfy):
    _, ages, _ = _families()
    box, conversation_id = _box(fake_comfy, ages)

    result, _ = box.run("run_workflow", json.dumps({"workflow": "krea-2/txt2img"}))

    assert result["run"] == 1
    assert conversations.get(conversation_id)["family_id"] == ages["id"]


def test_swapping_the_base_model_in_the_changes_counts(fake_comfy):
    krea, _, illustrious = _families()
    box, conversation_id = _box(fake_comfy, krea)

    result, _ = box.run("run_workflow", json.dumps({"workflow": "krea-2/txt2img", "changes": {
        "set": [{"node": "55", "input": "unet_name", "value": "illustrious\\some-model.safetensors"}]}}))

    assert result["family"] == "Illustrious"
    assert conversations.get(conversation_id)["family_id"] == illustrious["id"]


def test_the_library_shows_each_workflows_family():
    result = chat.build_system_prompt(
        {"instructions": "rules"},
        workflows=[{"name": "illustrious/txt2img", "description": "Tags.", "family": "Illustrious"},
                   {"name": "mine/loose", "description": "", "family": None}],
    )

    assert "- illustrious/txt2img [Illustrious]: Tags.\n- mine/loose\n" in result


def test_family_rules_handed_over_in_a_tool_result_are_not_replayed_later():
    steps = [{"content": "", "tool_calls": [{"id": "c1", "name": "run_workflow", "arguments": "{}"}],
              "results": [json.dumps({"error": "Not run yet", "family_rules": "long rules"})]},
             {"content": "Hecho.", "tool_calls": [], "results": []}]
    conversation = {"id": "c", "messages": [{"role": "user", "content": "x", "images": []},
                                            {"role": "assistant", "content": "Hecho.", "steps": steps},
                                            {"role": "user", "content": "otra", "images": []}]}

    messages = chat.build_llm_messages(conversation, lambda image: None, tools=True)

    assert "long rules" not in json.dumps(messages)


def test_the_reply_is_kept_by_the_family_it_ended_in(api_client, auth_headers, tmp_path, fake_comfy):
    krea, _, illustrious = _families()
    workflows.ensure_seeded()
    conversation_id = api_client.post("/api/conversations", auth=auth_headers,
                                      json={"family_id": krea["id"], "llm_model": "m"}).json()["id"]
    run = {"id": "c1", "name": "run_workflow", "arguments": json.dumps({
        "workflow": "illustrious/txt2img", "changes": {"positive": "1girl, witch", "negative": "lowres"}})}
    script = Script(("Pruebo con Illustrious.", [run]),
                    ("<prompt>\nPOSITIVE: 1girl, witch\nNEGATIVE: lowres\n</prompt>", [{**run, "id": "c2"}]))

    with patch("app.conversation_turn.stream_chat", script):
        api_client.post(f"/api/conversations/{conversation_id}/messages", auth=auth_headers, data={"content": "va"})

    # Step 2 got Illustrious' rules in the result of step 1 (a family switch isn't a launch:
    # the reply goes on); the launch in step 2 ends it.
    assert json.loads(script.calls[1]["messages"][-1]["content"])["family_rules"] == "Write danbooru tags."
    assert len(script.calls) == 2
    message = conversations.get(conversation_id)["messages"][-1]
    assert [p["type"] for p in message["parts"]] == ["text", "family", "prompt", "run"]
    # Illustrious uses negatives: the card keeps it (Krea 2 would have dropped it).
    assert next(p for p in message["parts"] if p["type"] == "prompt")["negative_prompt"] == "lowres"
    [entry] = history.list_entries()
    assert entry["family_name"] == "Illustrious"
    assert conversations.get(conversation_id)["family_id"] == illustrious["id"]
    # Next turn, the system prompt carries Illustrious' rules.
    with patch("app.conversation_turn.stream_chat", Script(("Vale.", []))) as next_turn:
        api_client.post(f"/api/conversations/{conversation_id}/messages", auth=auth_headers, data={"content": "y"})
    assert "Write danbooru tags." in next_turn.calls[0]["messages"][0]["content"]


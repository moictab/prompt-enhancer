"""The bot's view of the installed models is current, and a Civitai image's resources say
which installed file each one is -- even renamed."""
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

from app import catalog, conversations, families, image_metadata
from app.openrouter_client import StreamEnd

PEARLY = "anima\\pearly-sapphire_v1.0.safetensors"
ENTRIES = {
    PEARLY: {"folder": "diffusion_models", "family": "anima"},
    "krea-2\\style-aesthetic-quality-modifiers_v5.1.safetensors": {"folder": "loras", "family": "krea-2"},
    "krea-2\\style-ink_v1.safetensors": {"folder": "loras", "family": "krea-2"},
    "illustrious\\style-ink_v1.safetensors": {"folder": "loras", "family": "illustrious"},
}


def test_a_renamed_file_is_matched_by_its_file_name():
    resource = {"type": "checkpoint", "name": "Pearly Sapphire", "file": "pearlySapphire_v10.safetensors",
                "base_model": "Anima"}

    assert catalog.match_installed(resource, ENTRIES) == PEARLY


def test_or_by_most_of_the_words_of_its_name():
    assert catalog.match_installed({"type": "checkpoint", "name": "Pearly Sapphire"}, ENTRIES) == PEARLY
    # "Masterpiece LORA" isn't in the file name, but 3 of its 4 telling words are.
    aesthetic = {"type": "lora", "name": "Aesthetic Quality Modifiers - Masterpiece LORA", "file": "krea2-mp-v51.safetensors"}
    assert catalog.match_installed(aesthetic, ENTRIES) == "krea-2\\style-aesthetic-quality-modifiers_v5.1.safetensors"


def test_matching_respects_the_kind_of_file_and_the_base_model():
    # A LoRA isn't matched to a checkpoint, nor the other way around.
    assert catalog.match_installed({"type": "lora", "name": "Pearly Sapphire"}, ENTRIES) is None
    # Two files fit; the base model's folder decides.
    ink = {"type": "lora", "name": "Style Ink", "file": "style-ink_v1.safetensors", "base_model": "Illustrious"}
    assert catalog.match_installed(ink, ENTRIES) is None  # same file name in two folders: not clear by name
    ink_by_words = {"type": "lora", "name": "Ink Style Lines", "base_model": "Illustrious"}
    assert catalog.match_installed({**ink_by_words, "name": "style ink"}, ENTRIES) is None  # one telling word
    two_words = {"type": "lora", "name": "Ink Sketch Lines", "base_model": "Illustrious"}
    assert catalog.match_installed(two_words, ENTRIES) is None


def test_describe_says_which_resources_are_installed():
    metadata = {"format": "PNG", "width": 1, "height": 1, "generator": "Civitai", "resources": [
        {"type": "checkpoint", "name": "Pearly Sapphire", "file": "pearlySapphire_v10.safetensors"},
        {"type": "lora", "name": "Something Else Entirely"},
    ]}

    text = image_metadata.describe(metadata, 1, ENTRIES)

    assert f"installed here as {PEARLY}" in text
    assert 'lora "Something Else Entirely" (no installed file matches its name)' in text
    assert "installed here" not in image_metadata.describe(metadata, 1)  # no catalog: nothing claimed


def _stamp(**ago):
    return (datetime.now(timezone.utc) - timedelta(**ago)).isoformat()


def test_the_file_list_is_reread_when_a_couple_of_minutes_old(fake_comfy):
    catalog.refresh(fake_comfy)
    listed = catalog.load()["listed_at"]

    fake_comfy.model_files["diffusion_models"] = [PEARLY]
    catalog.ensure_fresh(fake_comfy)
    assert catalog.load()["listed_at"] == listed  # just listed: kept
    assert PEARLY not in catalog.load()["entries"]

    data = catalog.load()
    catalog._save({**data, "listed_at": _stamp(minutes=5)})
    with patch("app.catalog.refresh_in_background") as background:
        catalog.ensure_fresh(fake_comfy)

    entries = catalog.load()["entries"]
    assert entries[PEARLY] == {"folder": "diffusion_models", "family": "anima"}
    background.assert_not_called()  # no new LoRA: no metadata to read

    fake_comfy.model_files["loras"].append("anima\\new-style_v1.safetensors")
    catalog._save({**catalog.load(), "listed_at": _stamp(minutes=5)})
    with patch("app.catalog.refresh_in_background") as background:
        catalog.ensure_fresh(fake_comfy)
    background.assert_called_once()
    assert "checked" not in catalog.load()["entries"]["anima\\new-style_v1.safetensors"]


def test_a_comfyui_that_doesnt_answer_leaves_the_old_list(fake_comfy):
    catalog.refresh(fake_comfy)
    catalog._save({**catalog.load(), "listed_at": _stamp(minutes=5)})

    def stalled(folder, max_age=None, read_timeout=None):
        raise TimeoutError("stalled")

    fake_comfy.models = stalled
    catalog.ensure_fresh(fake_comfy)

    assert len(catalog.load()["entries"]) == 3


def test_a_turn_tells_the_bot_a_civitai_resource_is_installed(api_client, auth_headers, tmp_path, fake_comfy):
    family = families.create_family("Anima", "rules", False)
    conversation_id = api_client.post("/api/conversations", auth=auth_headers,
                                      json={"family_id": family["id"], "llm_model": "m"}).json()["id"]
    fake_comfy.model_files["diffusion_models"] = [PEARLY]
    metadata = {"format": "PNG", "width": 1, "height": 1, "generator": "Civitai",
                "resources": [{"type": "checkpoint", "name": "Pearly Sapphire", "base_model": "Anima"}]}
    from .conftest import png_bytes
    image = conversations.save_image(conversation_id, png_bytes(), "image/png", metadata=metadata)
    conversations.append_message(conversation_id, {"role": "user", "content": "mira", "images": [image]})
    sent = []

    def reply(api_key, model, messages, temperature, context=None, tools=None, tool_choice=None):
        sent.append(messages)
        yield "La tienes."
        yield StreamEnd(cost=None)

    with patch("app.conversation_turn.stream_chat", reply):
        api_client.post(f"/api/conversations/{conversation_id}/reply", auth=auth_headers)

    texts = [p["text"] for p in sent[0][1]["content"] if p["type"] == "text"]
    assert any(f"installed here as {PEARLY}" in text for text in texts)
    assert "pearly-sapphire_v1.0" in sent[0][0]["content"]  # and it is in the installed models list

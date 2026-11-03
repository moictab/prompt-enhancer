import json
from unittest.mock import patch

from app import catalog, conversations, families, workflows
from app.openrouter_client import StreamEnd

INK = "krea-2\\style-ink_v1.safetensors"
REAL = "krea-2\\base-real_v3.safetensors"
DETAIL = "illustrious\\detail_v2.safetensors"


def _refresh(fake_comfy):
    fake_comfy.model_files.update({
        "checkpoints": ["illustrious\\wai-v14.safetensors", "hunyuan_3d_v2.1.safetensors"],
        "diffusion_models": ["krea-2\\krea-2-turbo.safetensors"],
    })
    fake_comfy.model_metadata[INK] = {
        "ss_base_model_version": "krea2",
        "ss_tag_frequency": json.dumps({"1_ink": {"inkrot": 50, "linework": 30, "": 4, "monochrome": 10}}),
    }
    fake_comfy.model_metadata[DETAIL] = {"modelspec.trigger_phrase": "detailed, sharp", "modelspec.title": "Detail"}
    return catalog.refresh(fake_comfy)


def test_refresh_reads_files_and_lora_metadata(fake_comfy):
    data = _refresh(fake_comfy)

    entries = data["entries"]
    assert "hunyuan_3d_v2.1.safetensors" not in entries  # 3D, not images
    assert entries[INK]["base_model"] == "krea2"
    assert entries[INK]["triggers"] == ["inkrot", "linework", "monochrome"]
    assert entries[DETAIL]["triggers"] == ["detailed", "sharp"]
    assert entries[DETAIL]["title"] == "Detail"
    assert entries["illustrious\\wai-v14.safetensors"] == {"folder": "checkpoints", "family": "illustrious"}

    # Metadata is read once per file.
    fake_comfy.model_metadata[INK] = {"ss_base_model_version": "changed"}
    assert catalog.refresh(fake_comfy)["entries"][INK]["base_model"] == "krea2"


def test_descriptions_survive_refreshes(fake_comfy):
    _refresh(fake_comfy)
    catalog.set_description(INK, "  inky   linework, desaturates ")

    data = _refresh(fake_comfy)

    assert data["descriptions"][INK] == "inky linework, desaturates"
    assert catalog.details(INK)["description"] == "inky linework, desaturates"
    catalog.set_description(INK, "")
    assert "description" not in catalog.details(INK)


def test_prompt_section_groups_names_and_adds_what_is_known(fake_comfy):
    _refresh(fake_comfy)
    catalog.set_description(INK, "inky linework")

    section = catalog.prompt_section()

    assert section.splitlines()[1:] == [
        "Checkpoints:",
        "- illustrious: wai-v14",
        "Diffusion models:",
        "- krea-2: krea-2-turbo",
        "LoRAs:",
        "- illustrious: detail_v2",
        "- krea-2: base-real_v3, style-ink_v1 (inky linework)",
    ]


def test_prompt_section_is_none_before_the_first_refresh():
    assert catalog.prompt_section() is None


def test_stats_count_images_and_reactions_per_file(api_client, auth_headers, tmp_path, fake_comfy):
    family = families.create_family("Krea", "rules", False, path=str(tmp_path / "families.json"))
    conversation_id = api_client.post("/api/conversations", auth=auth_headers,
                                      json={"family_id": family["id"], "llm_model": "m"}).json()["id"]
    workflows.ensure_seeded()
    call = {"id": "c1", "name": "run_workflow", "arguments": json.dumps({
        "workflow": "krea-2/txt2img", "count": 2,
        # A sloppy name still lands on the installed file.
        "changes": {"loras": [{"name": "krea-2/STYLE-ink_v1", "strength": 0.4}]}})}
    replies = [("", [call]), ("Va.", []), ("Comentario.", [])]

    def fake(api_key, model, messages, temperature, context=None, tools=None, tool_choice=None):
        text, calls = replies.pop(0)
        if text:
            yield text
        yield StreamEnd(cost=None, tool_calls=calls)

    with patch("app.conversation_turn.stream_chat", fake):
        api_client.post(f"/api/conversations/{conversation_id}/messages", data={"content": "dos"}, auth=auth_headers)
        fake_comfy.finish("prompt-1", "a.png")
        fake_comfy.finish("prompt-2", "b.png")
        from app import runs
        runs.tracker.poll_once()
        runs.wait_until_idle()

    job = conversations.get_run(conversation_id, 1)["jobs"][0]
    assert job["loras"] == [{"name": INK, "strength": 0.4}]
    results = conversations.get(conversation_id)["messages"][2]
    first, second = (img["id"] for img in results["images"])
    put = f"/api/conversations/{conversation_id}/images"
    assert api_client.put(f"{put}/{first}/reaction", json={"rating": "up"}, auth=auth_headers).status_code == 200
    api_client.put(f"{put}/{second}/reaction", json={"rating": "down", "comment": "too inky"}, auth=auth_headers)

    assert catalog.stats()[INK] == {"images": 2, "liked": 1, "disliked": 1}
    assert catalog.details(INK)["results_here"] == {"images": 2, "liked": 1, "disliked": 1}


def test_admin_catalog_routes(api_client, auth_headers, fake_comfy):
    refreshed = api_client.post("/api/admin/catalog/refresh", auth=auth_headers)
    assert refreshed.status_code == 200
    assert {i["file"] for i in refreshed.json()["items"]} == {INK, REAL, DETAIL}

    described = api_client.put("/api/admin/catalog/description", auth=auth_headers,
                               json={"file": INK, "text": "inky"})
    assert described.json()["description"] == "inky"
    listed = api_client.get("/api/admin/catalog", auth=auth_headers).json()
    assert listed["configured"] is True
    assert next(i for i in listed["items"] if i["file"] == INK)["description"] == "inky"


def test_admin_refresh_without_comfy_is_400(api_client, auth_headers):
    assert api_client.post("/api/admin/catalog/refresh", auth=auth_headers).status_code == 400

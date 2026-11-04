"""Reference folders: folders of images the user accepted, which a message can point the bot at
("@name" or the path) for the app's study of them."""
import io
import json
from pathlib import Path
from unittest.mock import patch

import pytest
from PIL import Image

from app import chat, conversations, references

from .conftest import message_text, png_bytes
from .test_api_conversar_comfy import Script, _conversation, _events, _family, _send

GRAPH = {
    "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "illustrious\\wai-v14.safetensors"}},
    "2": {"class_type": "LoraLoader", "inputs": {"lora_name": "illustrious\\ramona_v2.safetensors",
                                                 "strength_model": 0.8, "strength_clip": 0.8,
                                                 "model": ["1", 0], "clip": ["1", 1]}},
    "3": {"class_type": "CLIPTextEncode", "inputs": {"text": "{positive}", "clip": ["2", 1]}},
    "4": {"class_type": "CLIPTextEncode", "inputs": {"text": "lowres, bad hands", "clip": ["2", 1]}},
    "5": {"class_type": "EmptyLatentImage", "inputs": {"width": 832, "height": 1216, "batch_size": 1}},
    "6": {"class_type": "KSampler", "inputs": {"seed": 1, "steps": 28, "cfg": 5, "sampler_name": "euler_ancestral",
                                               "scheduler": "normal", "denoise": 1, "model": ["2", 0],
                                               "positive": ["3", 0], "negative": ["4", 0], "latent_image": ["5", 0]}},
    "7": {"class_type": "VAEDecode", "inputs": {"samples": ["6", 0], "vae": ["1", 2]}},
    "8": {"class_type": "SaveImage", "inputs": {"images": ["7", 0], "filename_prefix": "x"}},
}


def comfy_png(positive: str, size=(83, 121)) -> bytes:
    graph = json.loads(json.dumps(GRAPH).replace("{positive}", positive))
    return png_bytes(size=size, prompt=graph)


def a1111_png(positive: str) -> bytes:
    return png_bytes(size=(80, 80), parameters=(
        f"{positive}, <lora:ink_style:0.6>\nNegative prompt: lowres, blurry\n"
        "Steps: 30, Sampler: DPM++ 2M, CFG scale: 6, Seed: 5, Size: 1024x1024, Model: waiIllustrious_v14"))


@pytest.fixture
def folder(tmp_path):
    root = tmp_path / "refs" / "Ramona"
    (root / "noche poses").mkdir(parents=True)
    (root / "a.png").write_bytes(comfy_png("1girl, red hair, freckles, green jacket, city street"))
    (root / "b.png").write_bytes(comfy_png("1girl, red hair, freckles, rooftop at dusk"))
    (root / "noche poses" / "c.png").write_bytes(comfy_png("1girl, red hair, freckles, neon night"))
    (root / "d.png").write_bytes(a1111_png("1girl, red hair, freckles, cafe"))
    out = io.BytesIO()
    Image.new("RGB", (60, 60), (10, 90, 10)).save(out, "JPEG")
    (root / "plain.jpg").write_bytes(out.getvalue())  # no metadata
    (root / "notes.txt").write_text("not an image")
    return root


def test_a_folder_is_accepted_by_its_full_path_and_once(folder):
    entry = references.create_folder("Ramona", str(folder))
    assert entry["path"] == str(folder.resolve())
    with pytest.raises(references.ReferenceFolderError, match="already a folder called"):
        references.create_folder("ramona", str(folder / "noche poses"))
    with pytest.raises(references.ReferenceFolderError, match="already accepted"):
        references.create_folder("Otra", str(folder))
    with pytest.raises(references.ReferenceFolderError, match="whole path"):
        references.create_folder("Rel", "refs/Ramona")
    with pytest.raises(references.ReferenceFolderError, match="no folder"):
        references.create_folder("Gone", str(folder / "missing"))
    counted = references.with_counts(entry)
    assert counted["images"] == 5 and not counted["missing"]


def test_a_message_points_at_a_folder_by_name_or_path(folder):
    entry = references.create_folder("Ramona", str(folder))
    folders = references.list_folders()
    assert references.find_mention("estudia @ramona, por favor", folders) == (entry, Path(entry["path"]))
    assert references.find_mention("escribe a ramona@example.com", folders) is None
    assert references.find_mention("mira Ramona", folders) is None
    path = str(folder)
    assert references.find_mention(f"mira {path}, por favor", folders)[1] == Path(entry["path"])
    found = references.find_mention(f'mira "{path}\\noche poses" y dime', folders)
    assert found[1] == (folder / "noche poses").resolve()
    assert references.find_mention(f"mira {path}\\noche poses\\..\\..\\.. ya", folders)[1] == Path(entry["path"])
    assert references.find_mention(f"mira {path}2", folders) is None
    assert references.find_mention(f"mira {path.replace(chr(92), '/')}/noche poses", folders)[1] == \
        (folder / "noche poses").resolve()


def test_the_study_says_what_recurs_and_shows_the_look(folder):
    entry = references.create_folder("Ramona", str(folder))
    installed = {"illustrious\\wai-v14.safetensors": {"folder": "checkpoints"}}
    found = references.study(entry, Path(entry["path"]), installed)

    assert found.counts == {"name": "Ramona", "path": entry["path"], "images": 5, "capped": False, "studied": 5,
                            "with_metadata": 4, "sheets": 1}
    text = found.text
    assert '"Ramona"' in text and "4 of them carry generation metadata" in text
    assert "wai-v14 (3; installed here as illustrious\\wai-v14.safetensors)" in text
    assert "ramona_v2 (3; weight 0.8" in text and "ink_style (1; weight 0.6" in text
    assert '"red hair" (4)' in text and '"freckles" (4)' in text and '"1girl" (4)' in text
    assert "euler_ancestral (3)" in text and "Made with: ComfyUI (3), A1111 (1), no metadata (1)" in text
    assert "2:3 portrait" in text
    assert "POSITIVE: 1girl, red hair, freckles" in text
    assert "R5 plain.jpg -- no metadata" in text or "plain.jpg -- no metadata" in text
    sheet = Image.open(io.BytesIO(found.sheets[0]))
    assert sheet.format == "JPEG" and sheet.size == (4 * references.TILE, 2 * references.TILE)
    # A folder inside it: only its images.
    inner = references.study(entry, (folder / "noche poses").resolve())
    assert inner.counts["images"] == 1 and "noche poses" in inner.text


def test_a_message_naming_a_folder_brings_its_study_once(api_client, auth_headers, tmp_path, folder):
    references.create_folder("Ramona", str(folder))
    conversation_id = _conversation(api_client, auth_headers, _family(tmp_path))["id"]
    script = Script(('Ramona: pelirroja con pecas en las 4.<remember>Ramona: red hair, freckles</remember>', []),
                    ("Vale.", []))

    with patch("app.conversation_turn.stream_chat", script):
        assert _events(_send(api_client, auth_headers, conversation_id, "estudia @Ramona"))[-1]["type"] == "done"
        _send(api_client, auth_headers, conversation_id, "¿y ahora?")

    stored = conversations.get(conversation_id)["messages"][0]
    assert stored["references"]["name"] == "Ramona" and stored["references"]["sheets"] == 1
    assert [img.get("sheet") for img in stored["images"]] == [True]
    first, second = (message_text(call["messages"][-1]) for call in script.calls)
    assert "Reference study" in first and '"red hair" (4)' in first
    # Answered: one line, not the whole study again.
    replayed = message_text(next(m for m in script.calls[1]["messages"] if m["role"] == "user"))
    assert "Reference study" not in replayed and "you answered it below" in replayed
    shown = api_client.get(f"/api/conversations/{conversation_id}", auth=auth_headers).json()["messages"][0]
    assert shown["reference_study"] is True


def test_a_folder_that_cant_be_studied_refuses_the_message(api_client, auth_headers, tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    references.create_folder("Vacia", str(empty))
    conversation_id = _conversation(api_client, auth_headers, _family(tmp_path))["id"]
    response = _send(api_client, auth_headers, conversation_id, "mira @Vacia")
    assert response.status_code == 400 and "no images" in response.json()["detail"]
    assert conversations.get(conversation_id)["messages"] == []


def test_ajustes_lists_adds_and_removes_folders(api_client, auth_headers, folder):
    created = api_client.post("/api/admin/reference-folders", auth=auth_headers,
                              json={"name": "Ramona", "path": str(folder)})
    assert created.status_code == 200 and created.json()["images"] == 5
    assert api_client.post("/api/admin/reference-folders", auth=auth_headers,
                           json={"name": "X", "path": "nope"}).status_code == 400
    listed = api_client.get("/api/admin/reference-folders", auth=auth_headers).json()
    assert [f["name"] for f in listed] == ["Ramona"]
    assert api_client.delete(f"/api/admin/reference-folders/{listed[0]['id']}", auth=auth_headers).status_code == 200
    assert api_client.get("/api/admin/reference-folders", auth=auth_headers).json() == []
    assert folder.joinpath("a.png").exists()  # nothing in it is touched

"""Downloads from Civitai: the bot prepares them, the user accepts each one, the app fetches a
SafeTensors file into ComfyUI's models folder under the owner's naming."""
import hashlib
import json
import struct
import time
from unittest.mock import patch

import pytest

from app import chat, conversations, downloads, workflows

from .conftest import message_text
from .test_api_conversar_comfy import Script, _call, _conversation, _family, _send

SAFETENSORS = struct.pack("<Q", 2) + b"{}" + b"\0" * 64
VERSION = {
    "id": 456, "modelId": 123, "name": "v1.0", "baseModel": "Illustrious",
    "model": {"name": "Ramona Flowers | Illustrious LoRA", "type": "LORA"},
    "files": [
        {"id": 1, "name": "ramona.pt", "type": "Model", "sizeKB": 1, "metadata": {"format": "PickleTensor"}},
        {"id": 2, "name": "RamonaFlowers_IL.safetensors", "type": "Model", "primary": True, "sizeKB": 0.07,
         "metadata": {"format": "SafeTensor"}, "hashes": {"SHA256": hashlib.sha256(SAFETENSORS).hexdigest().upper()}},
    ],
}
CHECKPOINT = {
    "id": 789, "modelId": 321, "name": "v14.0", "baseModel": "Illustrious",
    "model": {"name": "WAI-NSFW-illustrious-SDXL", "type": "Checkpoint"},
    "files": [{"id": 3, "name": "wai.safetensors", "type": "Model", "primary": True, "sizeKB": 0.07,
               "metadata": {"format": "SafeTensor"}, "hashes": {}}],
}


def test_files_get_the_owners_names():
    assert downloads.file_name("Ramona Flowers | Illustrious LoRA", "v1.0", "loras", ["character", "anime"]) == \
        "char-ramona-flowers_v1.0"
    assert downloads.file_name("Arcane Style [Pony/IL]", "V2", "loras", ["style"]) == "style-arcane-style_v2.0"
    assert downloads.file_name("Detail Tweaker XL", "Illustrious", "loras", []) == "concept-detail-tweaker"
    assert downloads.file_name("WAI-NSFW-illustrious-SDXL", "v14.0", "checkpoints") == "wai-nsfw_v14.0"
    assert downloads.file_name("Pikon-Realism", "v2 alt", "checkpoints") == "pikon-realism_v2.0"


@pytest.fixture
def models(tmp_path, monkeypatch):
    root = tmp_path / "models"
    for folder in ("loras/illustrious", "loras/krea-2", "checkpoints/illustrious", "diffusion_models/anima"):
        (root / folder).mkdir(parents=True)
    monkeypatch.setenv("COMFY_MODELS_DIR", str(root))
    return root


@pytest.fixture
def civitai_api(monkeypatch):
    answers = {"/model-versions/456": VERSION, "/model-versions/789": CHECKPOINT,
               "/models/123": {"tags": ["character", "anime"]}, "/model-versions/by-hash/ABC123": VERSION}
    asked = []

    def fake(path, params=None):
        asked.append(path)
        return answers.get(path)

    monkeypatch.setattr("app.civitai._get_json", fake)
    return asked


class FakeFile:
    def __init__(self, body=SAFETENSORS, status=200):
        self.body, self.status_code = body, status
        self.headers = {"Content-Length": str(len(body))}

    def iter_content(self, size):
        for i in range(0, len(self.body), 16):
            yield self.body[i:i + 16]

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _wait(conversation_id, n, timeout=5):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        record = conversations.get_download(conversation_id, n)
        if record["status"] != "downloading":
            return record
        time.sleep(0.02)
    raise AssertionError("still downloading")


def test_the_bot_prepares_and_only_the_user_starts_it(api_client, auth_headers, tmp_path, fake_comfy, models,
                                                     civitai_api):
    workflows.ensure_seeded()
    conversation_id = _conversation(api_client, auth_headers, _family(tmp_path))["id"]
    script = Script(("Te preparo la descarga del LoRA de Ramona:", [_call("prepare_download", version_id=456)]),
                    ("Listo: dale a Descargar si lo quieres.", []), ("Ya lo tienes: lo pruebo.", []), ("Vale.", []))
    opened = []

    with patch("app.conversation_turn.stream_chat", script):
        _send(api_client, auth_headers, conversation_id, "quiero el LoRA de Ramona")
        record = conversations.get_download(conversation_id, 1)
        assert record["status"] == "proposed" and record["folder"] == "loras" and record["family"] == "illustrious"
        assert record["name"] == "char-ramona-flowers_v1.0" and record["civitai_file"] == "RamonaFlowers_IL.safetensors"
        assert not list(models.rglob("*.safetensors"))  # nothing until the user accepts
        reply = conversations.get(conversation_id)["messages"][-1]
        assert {"type": "download", "download": 1} in reply["parts"]

        def fake_open(url):
            opened.append(url)
            return FakeFile()

        with patch("app.downloads._open", fake_open):
            accepted = api_client.post(f"/api/conversations/{conversation_id}/downloads/1/accept", auth=auth_headers,
                                       json={"name": "char-ramona_v1.0"})
            assert accepted.status_code == 200 and accepted.json()["status"] == "downloading"
            done = _wait(conversation_id, 1)
        assert done["status"] == "done" and done["installed_as"] == "illustrious\\char-ramona_v1.0.safetensors"
        assert (models / "loras" / "illustrious" / "char-ramona_v1.0.safetensors").read_bytes() == SAFETENSORS
        assert opened == ["https://civitai.com/api/download/models/456?type=Model&format=SafeTensor"]
        assert not list(models.rglob("*.part"))
        # The bot hears of it once, in the next turn's context.
        _send(api_client, auth_headers, conversation_id, "¿y ahora?")
        _send(api_client, auth_headers, conversation_id, "vale")
    told, later = (message_text(call["messages"][-1]) for call in script.calls[2:])
    assert "the user downloaded it; installed as loras\\illustrious\\char-ramona_v1.0.safetensors" in told
    assert "Downloads Since Your Last Reply" not in later


def test_it_never_overwrites_and_never_takes_a_pickle(api_client, auth_headers, tmp_path, models, civitai_api,
                                                       monkeypatch):
    conversation_id = _conversation(api_client, auth_headers, _family(tmp_path))["id"]
    record = downloads.prepare(conversation_id, {"version_id": 456})
    (models / "loras" / "illustrious" / "char-ramona-flowers_v1.0.safetensors").write_bytes(b"mine")
    with pytest.raises(downloads.DownloadError, match="already"):
        downloads.accept(conversation_id, record["n"])
    with pytest.raises(downloads.DownloadError, match="no family folder"):
        downloads.accept(conversation_id, record["n"], family="..\\..")
    with pytest.raises(downloads.DownloadError, match="lowercase"):
        downloads.accept(conversation_id, record["n"], name="..\\evil")
    with pytest.raises(downloads.DownloadError, match="goes in loras"):
        downloads.accept(conversation_id, record["n"], folder="checkpoints")
    assert (models / "loras" / "illustrious" / "char-ramona-flowers_v1.0.safetensors").read_bytes() == b"mine"

    pickle_only = {**VERSION, "id": 457, "files": VERSION["files"][:1]}
    monkeypatch.setattr("app.civitai._get_json", lambda path, params=None: pickle_only)
    with pytest.raises(downloads.DownloadError, match="no SafeTensors file"):
        downloads.prepare(conversation_id, {"version_id": 457})


def test_a_file_that_isnt_what_civitai_said_is_thrown_away(api_client, auth_headers, tmp_path, models, civitai_api):
    conversation_id = _conversation(api_client, auth_headers, _family(tmp_path))["id"]
    record = downloads.prepare(conversation_id, {"url": "https://civitai.com/models/123?modelVersionId=456"})
    with patch("app.downloads._open", lambda url: FakeFile(SAFETENSORS + b"tampered")):
        downloads.accept(conversation_id, record["n"])
        failed = _wait(conversation_id, record["n"])
    assert failed["status"] == "failed" and "checksum" in failed["error"]
    assert not list(models.rglob("*.safetensors*"))
    with patch("app.downloads._open", lambda url: FakeFile(b"<html>sign in</html>" + b"\0" * 64)):
        downloads.accept(conversation_id, record["n"])  # a failed one can be accepted again
        assert "isn't a SafeTensors file" in _wait(conversation_id, record["n"])["error"]
    with patch("app.downloads._open", lambda url: FakeFile(status=401)):
        downloads.accept(conversation_id, record["n"])
        assert "sign in" in _wait(conversation_id, record["n"])["error"]


def test_a_checkpoint_goes_to_its_family_and_can_be_turned_down(api_client, auth_headers, tmp_path, models,
                                                                civitai_api):
    conversation_id = _conversation(api_client, auth_headers, _family(tmp_path))["id"]
    record = downloads.prepare(conversation_id, {"hash": "ABC123"})
    assert record["version_id"] == 456  # found by its hash
    checkpoint = downloads.prepare(conversation_id, {"version_id": 789})
    assert (checkpoint["folder"], checkpoint["family"], checkpoint["name"]) == \
        ("checkpoints", "illustrious", "wai-nsfw_v14.0")
    assert downloads.prepare(conversation_id, {"version_id": 789})["n"] == checkpoint["n"]  # prepared once
    declined = api_client.post(f"/api/conversations/{conversation_id}/downloads/{checkpoint['n']}/decline",
                               auth=auth_headers)
    assert declined.json()["status"] == "declined"
    assert chat.turn_context([], downloads=downloads.news_text([declined.json()])).count("turned it down") == 1
    folders = api_client.get("/api/model-folders", auth=auth_headers).json()
    assert folders["loras"] == ["illustrious", "krea-2"] and folders["diffusion_models"] == ["anima"]


def test_without_a_models_folder_there_are_no_downloads(api_client, auth_headers, tmp_path, civitai_api):
    conversation_id = _conversation(api_client, auth_headers, _family(tmp_path))["id"]
    with pytest.raises(downloads.DownloadError, match="COMFY_MODELS_DIR"):
        downloads.prepare(conversation_id, {"version_id": 456})
    assert api_client.get("/api/model-folders", auth=auth_headers).json() == {}
    assert json.dumps(civitai_api) == "[]"  # Civitai wasn't even asked

import io
import json

import pytest
from fastapi.testclient import TestClient
from PIL import Image
from PIL.PngImagePlugin import PngInfo

from app import catalog, comfy, runs
from app.main import app


@pytest.fixture(autouse=True)
def isolated_data_dir(tmp_path, monkeypatch):
    # Every OpenRouter call writes to the request log under DATA_DIR, including in
    # tests that never build the app: keep all of it out of the repo's data/.
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    # ComfyUI runs are off unless a test turns them on (a developer's .env may set these).
    monkeypatch.delenv("COMFY_GATEWAY_URL", raising=False)
    monkeypatch.delenv("COMFY_DIRECT_URL", raising=False)
    monkeypatch.delenv("CIVITAI_API_KEY", raising=False)
    monkeypatch.delenv("COMFY_MODELS_DIR", raising=False)  # downloads write there: never the real one
    # The app runs from this checkout, so its .env holds the real key: no test may spend it.
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setenv("ADMIN_PASSWORD", "test-password")
    runs.tracker._pending.clear()
    catalog.invalidate_stats()
    yield
    runs.wait_until_idle()
    runs.tracker._pending.clear()


@pytest.fixture
def api_client(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("ADMIN_USERNAME", "admin")
    monkeypatch.setenv("ADMIN_PASSWORD", "test-password")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    return TestClient(app)


@pytest.fixture
def auth_headers():
    return ("admin", "test-password")


def png_bytes(size=(64, 48), color=(200, 30, 30), **texts) -> bytes:
    """A real PNG, with text chunks (e.g. prompt=json.dumps(graph), parameters=...)."""
    info = PngInfo()
    for key, value in texts.items():
        info.add_text(key, value if isinstance(value, str) else json.dumps(value))
    out = io.BytesIO()
    Image.new("RGB", size, color).save(out, "PNG", pnginfo=info)
    return out.getvalue()


class FakeComfy:
    """Stands in for comfy.Comfy: records what is sent, answers from what tests set."""

    base = "http://gateway"

    def __init__(self):
        self.submitted: list[dict] = []
        self.priorities: list[str] = []
        self.tags: list[str] = []
        self.histories: dict[str, dict] = {}
        self.files: dict[str, bytes] = {}
        self.uploads: list[str] = []
        self.cancelled: list[str] = []
        self.model_files = {"loras": ["krea-2\\style-ink_v1.safetensors", "krea-2\\base-real_v3.safetensors",
                                      "illustrious\\detail_v2.safetensors"]}
        self.model_metadata: dict[str, dict] = {}
        self.node_definitions: dict[str, dict] = {}  # /object_info entries, by class type
        self.reject: Exception | None = None
        self.unreachable = False

    def pick_base(self):
        if self.unreachable:
            raise comfy.Unreachable("ComfyUI isn't reachable: the PC may be off.")
        return self.base

    def submit(self, base, workflow, tag="Conversar", priority="high"):
        if self.reject is not None:
            raise self.reject
        self.submitted.append(workflow)
        self.priorities.append(priority)
        self.tags.append(tag)
        return f"prompt-{len(self.submitted)}"

    def history(self, base, prompt_id):
        return self.histories.get(prompt_id, {})

    def view(self, base, filename, subfolder="", type_="output"):
        return self.files[filename], "image/png"

    def upload_image(self, base, data, filename, content_type):
        self.uploads.append(filename)
        return f"{comfy.UPLOAD_SUBFOLDER}/{filename}"

    def cancel(self, base, prompt_id):
        self.cancelled.append(prompt_id)

    def models(self, folder, max_age=None, read_timeout=None):
        return self.model_files.get(folder, [])

    def metadata(self, folder, filename):
        return self.model_metadata.get(filename)

    def node_info(self, class_type):
        return self.node_definitions.get(class_type)

    def finish(self, prompt_id, filename=None, status="success", error=None, seconds=None):
        """Make a job done: its image (a real PNG) appears in /history and /view; seconds: how long
        ComfyUI took (its execution start and end messages)."""
        outputs = {}
        if filename:
            self.files[filename] = png_bytes(color=(10, 120, 200))
            outputs = {"29": {"images": [{"filename": filename, "subfolder": "", "type": "output"}]}}
        status_entry = {"status_str": status, "completed": status == "success", "messages": []}
        if seconds is not None:
            status_entry["messages"] = [["execution_start", {"prompt_id": prompt_id, "timestamp": 1_000_000}],
                                        ["execution_success", {"prompt_id": prompt_id,
                                                               "timestamp": 1_000_000 + seconds * 1000}]]
        if error:
            status_entry["messages"] = [["execution_error", {"exception_type": "RuntimeError",
                                                              "exception_message": error, "node_type": "KSampler"}]]
        self.histories[prompt_id] = {"outputs": outputs, "status": status_entry}


@pytest.fixture
def fake_comfy(monkeypatch):
    fake = FakeComfy()
    monkeypatch.setattr(comfy, "client", lambda: fake)
    monkeypatch.setattr("app.conversation_turn._model_takes_tools", lambda model: True)
    monkeypatch.setattr("app.catalog.refresh_in_background", lambda client, force=False: False)
    return fake


def message_text(message: dict) -> str:
    """A chat message's text, whether its content is a string or a list of parts."""
    content = message.get("content") or ""
    if isinstance(content, str):
        return content
    return "\n".join(part.get("text", "") for part in content if part.get("type") == "text")

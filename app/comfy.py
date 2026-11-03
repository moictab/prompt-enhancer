"""The owner's ComfyUI, reached through comfy-gateway (same API as ComfyUI, plus a queue
and an arbiter in front of it; rules in comfyui-manager/docs/integracion.md).

- The URL is chosen per run: the gateway, or ComfyUI directly only when the gateway
  doesn't answer the connection (a 4xx/5xx is an answer). Everything for that run --
  upload, /prompt, /history, /view, cancelling -- uses the same URL.
- Project, priority and tag go in extra_data.gateway (ComfyUI directly ignores them).
- Runs are followed by polling /history/{prompt_id}; runs.py does that.
"""
import logging
import time
import uuid
from dataclasses import dataclass

import requests

from .config import get_settings

logger = logging.getLogger("app.comfy")

PROJECT = "prompt-enhancer"
# Where this app's files live in ComfyUI: uploads in input/<UPLOAD_SUBFOLDER>/, images it makes
# in output/<OUTPUT_SUBFOLDER>/. It reads nothing else back from ComfyUI (see comfy_policy).
UPLOAD_SUBFOLDER = "prompt-enhancer"
OUTPUT_SUBFOLDER = "prompt-enhancer"
OUTPUT_PREFIX = f"{OUTPUT_SUBFOLDER}/conversar"
MODEL_FOLDERS = ("checkpoints", "diffusion_models", "loras", "vae", "text_encoders", "upscale_models",
                 "controlnet", "embeddings")
PRIORITY = "high"  # a person is waiting for the result in the chat
CONNECT_TIMEOUT = 4
READ_TIMEOUT = 60
MODELS_CACHE_SECONDS = 300
BASE_CACHE_SECONDS = 30
CONNECT_ERRORS = (requests.exceptions.ConnectionError, requests.exceptions.ConnectTimeout)


class Unreachable(RuntimeError):
    pass


class ComfyError(RuntimeError):
    """ComfyUI (or the gateway) answered with an error, e.g. a workflow that fails validation."""

    def __init__(self, status: int, message: str, node_errors: dict | None = None):
        super().__init__(message)
        self.status = status
        self.node_errors = node_errors or {}


@dataclass
class Result:
    status: str  # "success" | "error" | "interrupted"
    outputs: list[dict]
    error: str | None = None


def outputs_from_history(entry: dict) -> list[dict]:
    """The saved images of a finished job; previews only when nothing was saved."""
    saved, previews = [], []
    for node_id, out in (entry.get("outputs") or {}).items():
        for image in out.get("images") or []:
            if not isinstance(image, dict) or not image.get("filename"):
                continue
            item = {"node": node_id, "filename": image["filename"],
                    "subfolder": image.get("subfolder", ""), "type": image.get("type", "output")}
            (saved if item["type"] == "output" else previews).append(item)
    return saved or previews


def _error_text(message) -> str | None:
    if not isinstance(message, dict):
        return None
    parts = [message.get("exception_type"), message.get("exception_message")]
    node = message.get("node_type")
    text = ": ".join(str(p).strip() for p in parts if p)
    return f"{text} (in {node})" if node and text else text or None


def result_from_history(entry: dict) -> Result | None:
    """The result of a /history entry, or None while the job hasn't finished."""
    if not entry:
        return None
    status = entry.get("status") or {}
    if not status.get("completed") and status.get("status_str") not in ("error", "success"):
        return None
    if status.get("status_str") == "error":
        messages = status.get("messages") or []
        error = next((_error_text(m[1]) for m in messages if m and m[0] == "execution_error"), None)
        interrupted = any(m and m[0] == "execution_interrupted" for m in messages)
        return Result("interrupted" if interrupted else "error", outputs_from_history(entry),
                      error or ("interrupted" if interrupted else "ComfyUI reported an error"))
    return Result("success", outputs_from_history(entry))


class Comfy:
    def __init__(self, gateway_url: str, direct_url: str = "", project: str = PROJECT):
        self.gateway_url = gateway_url.rstrip("/")
        self.direct_url = direct_url.rstrip("/")
        self.project = project
        self.client_id = uuid.uuid4().hex
        self._base: tuple[str, float] | None = None
        self._models: dict[str, tuple[list[str], float]] = {}

    def _bases(self) -> list[str]:
        return [b for b in (self.gateway_url, self.direct_url) if b]

    def _request(self, method: str, url: str, **kwargs) -> requests.Response:
        kwargs.setdefault("timeout", (CONNECT_TIMEOUT, READ_TIMEOUT))
        response = requests.request(method, url, **kwargs)
        if response.status_code >= 400:
            try:
                body = response.json()
            except ValueError:
                body = {}
            error = body.get("error") if isinstance(body, dict) else None
            message = (error.get("message") if isinstance(error, dict) else error) or response.text[:500]
            node_errors = body.get("node_errors") if isinstance(body, dict) else None
            raise ComfyError(response.status_code, str(message), node_errors)
        return response

    def pick_base(self) -> str:
        """The gateway if it answers, else ComfyUI directly (logged); remembered for a little while."""
        if self._base and time.monotonic() - self._base[1] < BASE_CACHE_SECONDS:
            return self._base[0]
        for base in self._bases():
            # Only the connection matters here: a busy ComfyUI can take half a minute to answer
            # /system_stats, but if it accepted the connection, it is there.
            try:
                self._request("GET", f"{base}/models", timeout=(CONNECT_TIMEOUT, 1.5))
            except requests.exceptions.ConnectTimeout:
                logger.warning("comfy not answering base=%s", base)
                continue
            except requests.exceptions.ReadTimeout:
                pass
            except requests.exceptions.ConnectionError:
                logger.warning("comfy not answering base=%s", base)
                continue
            except (ComfyError, requests.exceptions.RequestException):
                pass  # it answered: that's what matters here
            if base != self.gateway_url:
                logger.warning("comfy gateway not answering, using ComfyUI directly base=%s", base)
            self._base = (base, time.monotonic())
            return base
        self._base = None
        raise Unreachable("ComfyUI isn't reachable (neither the gateway nor ComfyUI answer): the PC may be off.")

    def submit(self, base: str, workflow: dict, tag: str = "conversar") -> str:
        body = {
            "prompt": workflow,
            "client_id": self.client_id,
            "extra_data": {"gateway": {"project": self.project, "priority": PRIORITY, "tag": tag}},
        }
        try:
            response = self._request("POST", f"{base}/prompt", json=body)
        except CONNECT_ERRORS:
            self._base = None
            raise Unreachable(f"ComfyUI stopped answering ({base}).")
        prompt_id = response.json().get("prompt_id")
        if not prompt_id:
            raise ComfyError(response.status_code, f"ComfyUI accepted the workflow but returned no prompt_id: "
                                                   f"{response.text[:200]}")
        logger.info("comfy submitted base=%s prompt_id=%s nodes=%d", base, prompt_id, len(workflow))
        return prompt_id

    def history(self, base: str, prompt_id: str) -> dict:
        response = self._request("GET", f"{base}/history/{prompt_id}", timeout=(CONNECT_TIMEOUT, 20))
        return (response.json() or {}).get(prompt_id) or {}

    def view(self, base: str, filename: str, subfolder: str = "", type_: str = "output") -> tuple[bytes, str]:
        """An image one of this app's jobs made: saved under OUTPUT_SUBFOLDER, or a preview
        (temp) listed in its own /history. Nothing from the input folder, nothing else saved."""
        own_output = type_ == "output" and (subfolder == OUTPUT_SUBFOLDER or subfolder.startswith(f"{OUTPUT_SUBFOLDER}/"))
        if not (own_output or type_ == "temp"):
            raise ComfyError(403, f"Not an image this app made: {type_}/{subfolder}/{filename}")
        response = self._request("GET", f"{base}/view",
                                 params={"filename": filename, "subfolder": subfolder, "type": type_})
        return response.content, response.headers.get("Content-Type", "").split(";")[0].strip()

    def upload_image(self, base: str, data: bytes, filename: str, content_type: str) -> str:
        """Upload an input image into UPLOAD_SUBFOLDER; returns the name a LoadImage node takes."""
        response = self._request("POST", f"{base}/upload/image", files={"image": (filename, data, content_type)},
                                 data={"overwrite": "true", "subfolder": UPLOAD_SUBFOLDER})
        info = response.json()
        return f"{info['subfolder']}/{info['name']}" if info.get("subfolder") else info["name"]

    def cancel(self, base: str, prompt_id: str) -> None:
        """Take it off the queue if waiting, and stop it if running."""
        self._request("POST", f"{base}/queue", json={"delete": [prompt_id]})
        self._request("POST", f"{base}/interrupt", json={"prompt_id": prompt_id})

    def metadata(self, folder: str, filename: str) -> dict | None:
        """A model file's own metadata (the .safetensors header), or None if it has none."""
        if folder not in MODEL_FOLDERS:
            raise ValueError(f"not a model folder: {folder}")
        base = self.pick_base()
        response = self._request("GET", f"{base}/view_metadata/{folder}",
                                 params={"filename": filename.replace("\\", "/")}, timeout=(CONNECT_TIMEOUT, 60))
        try:
            data = response.json()
        except ValueError:
            return None
        return data if isinstance(data, dict) else None

    def models(self, folder: str) -> list[str]:
        if folder not in MODEL_FOLDERS:
            raise ValueError(f"not a model folder: {folder}")
        cached = self._models.get(folder)
        if cached and time.monotonic() - cached[1] < MODELS_CACHE_SECONDS:
            return cached[0]
        base = self.pick_base()
        names = self._request("GET", f"{base}/models/{folder}", timeout=(CONNECT_TIMEOUT, 60)).json()
        names = [n for n in names if isinstance(n, str)] if isinstance(names, list) else []
        self._models[folder] = (names, time.monotonic())
        return names


_client: dict = {}


def client() -> Comfy | None:
    """The configured ComfyUI client, or None when COMFY_GATEWAY_URL/COMFY_DIRECT_URL are unset."""
    settings = get_settings()
    if not (settings.comfy_gateway_url or settings.comfy_direct_url):
        return None
    key = (settings.comfy_gateway_url, settings.comfy_direct_url)
    if _client.get("key") != key:
        gateway = settings.comfy_gateway_url or settings.comfy_direct_url
        direct = settings.comfy_direct_url if settings.comfy_gateway_url else ""
        _client.update(key=key, comfy=Comfy(gateway, direct))
    return _client["comfy"]

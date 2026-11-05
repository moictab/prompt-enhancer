"""The owner's ComfyUI, reached through comfy-gateway (same API as ComfyUI, plus a queue
and an arbiter in front of it; rules in comfyui-manager/docs/integracion.md).

- The URL is chosen per run: the gateway, or ComfyUI directly only when the gateway
  doesn't answer the connection (a 4xx/5xx is an answer). Everything for that run --
  upload, /prompt, /history, /view, cancelling -- uses the same URL.
- Project, priority and tag go in extra_data.gateway (ComfyUI directly ignores them).
- Runs are followed by polling /history/{prompt_id}; runs.py does that.
"""
import logging
import re
import time
import uuid
from dataclasses import dataclass
from urllib.parse import quote

import requests

from .config import get_settings

logger = logging.getLogger("app.comfy")

PROJECT = "prompt-enhancer"
# Where this app's files live in ComfyUI: uploads in input/<UPLOAD_SUBFOLDER>/, images it makes
# in output/<OUTPUT_SUBFOLDER>/<one folder per conversation>/. It reads nothing else back from
# ComfyUI (see comfy_policy).
UPLOAD_SUBFOLDER = "prompt-enhancer"
OUTPUT_SUBFOLDER = "prompt-enhancer"
MODEL_FOLDERS = ("checkpoints", "diffusion_models", "loras", "vae", "text_encoders", "upscale_models",
                 "controlnet", "embeddings")
# Priorities in comfy-gateway's queue (idle | low | normal | high | urgent; within a level, a project's
# jobs go in the order sent, so a big batch would hold up a couple of images from another conversation).
PRIORITY = "high"  # the first images of a reply the user asked for: they are waiting for them
PRIORITY_REST = "normal"  # the rest of that reply's images (an exploring batch)
# What the bot launches by itself, looking at its own results: no one is waiting for it right now.
PRIORITY_ON_ITS_OWN = "low"
# Finishing an image (finish_image), even one the user just loved, and also when the bot does it on
# its own: 4-5 minutes of GPU each, so at "high" three of them spent the project's 15 minutes an hour
# there, and the drafts the user waits for dropped to "normal". Not "low" either: the gateway stops a
# low job for a high one and runs it again from the start -- minutes of GPU thrown away each time; it
# never stops a normal one. "Pasar delante" is there for one wanted sooner.
PRIORITY_FINISH = "normal"
# A run the user put ahead of everything else waiting ("Pasar delante"): the gateway owner's override,
# which skips the hourly limits on urgent and high (Comfy.prioritize).
PRIORITY_AHEAD = "urgent"
CONNECT_TIMEOUT = 4
READ_TIMEOUT = 60
MODELS_CACHE_SECONDS = 300
NODE_INFO_CACHE_SECONDS = 3600
BASE_CACHE_SECONDS = 30
CONNECT_ERRORS = (requests.exceptions.ConnectionError, requests.exceptions.ConnectTimeout)


def output_prefix(folder: str, run: int, seed=None) -> str:
    """ComfyUI's filename_prefix for one job of a run: its images are saved as
    output/prompt-enhancer/<folder>/run-03_seed-123_00001_.png."""
    name = f"run-{run:02d}" + (f"_seed-{seed}" if isinstance(seed, int) else "")
    return f"{OUTPUT_SUBFOLDER}/{folder}/{name}"


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
    seconds: float | None = None  # how long ComfyUI took making it, queue wait aside


def execution_seconds(messages) -> float | None:
    """From a /history entry's status messages: its last execution start to its end, in seconds."""
    start = end = None
    for message in messages or []:
        if not (isinstance(message, (list, tuple)) and len(message) == 2 and isinstance(message[1], dict)):
            continue
        kind, stamp = message[0], message[1].get("timestamp")
        if not isinstance(stamp, (int, float)):
            continue
        if kind == "execution_start":
            start = stamp
        elif kind in ("execution_success", "execution_error", "execution_interrupted"):
            end = stamp
    if start is None or end is None or end < start:
        return None
    return round((end - start) / 1000, 1)


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
    messages = status.get("messages") or []
    if status.get("status_str") == "error":
        error = next((_error_text(m[1]) for m in messages if m and m[0] == "execution_error"), None)
        interrupted = any(m and m[0] == "execution_interrupted" for m in messages)
        return Result("interrupted" if interrupted else "error", outputs_from_history(entry),
                      error or ("interrupted" if interrupted else "ComfyUI reported an error"),
                      execution_seconds(messages))
    return Result("success", outputs_from_history(entry), seconds=execution_seconds(messages))


def result_from_gateway(record: dict | None) -> Result | None:
    """The result the gateway noted for a job that finished well -- the images it saved, from
    ComfyUI's messages as they came -- for when ComfyUI's /history has forgotten it (a restart).
    None for anything else: the gateway notes no error details, and those /history keeps anyway."""
    if not record or record.get("status") != "success":
        return None
    outputs = [{"node": "", "filename": o["filename"], "subfolder": o.get("subfolder") or "", "type": "output"}
               for o in record.get("outputs") or []
               if isinstance(o, dict) and o.get("filename") and o.get("type") == "output"]  # "temp": previews
    if not outputs:
        return None
    started, finished = record.get("started_at"), record.get("finished_at")
    seconds = finished - started if all(isinstance(t, (int, float)) for t in (started, finished)) else None
    return Result("success", outputs, seconds=seconds)


# Comfy.gateway_job when the gateway can't be asked right now (down, restarting, a bad answer).
GATEWAY_UNKNOWN = {"status": "unknown"}


class Comfy:
    def __init__(self, gateway_url: str, direct_url: str = "", project: str = PROJECT):
        self.gateway_url = gateway_url.rstrip("/")
        self.direct_url = direct_url.rstrip("/")
        self.project = project
        self.client_id = uuid.uuid4().hex
        self._base: tuple[str, float] | None = None
        self._models: dict[str, tuple[list[str], float]] = {}
        self._nodes: dict[str, tuple[dict | None, float]] = {}

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

    def submit(self, base: str, workflow: dict, tag: str = "Conversar", priority: str = PRIORITY) -> str:
        body = {
            "prompt": workflow,
            "client_id": self.client_id,
            "extra_data": {"gateway": {"project": self.project, "priority": priority, "tag": tag}},
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
        # ComfyUI on Windows reports subfolders with backslashes.
        parts = [part for part in re.split(r"[\\/]", subfolder or "") if part]
        own_output = type_ == "output" and parts[:1] == [OUTPUT_SUBFOLDER] and ".." not in parts
        if not (own_output or (type_ == "temp" and ".." not in parts)):
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

    def prioritize(self, base: str, prompt_id: str, priority: str = PRIORITY_AHEAD) -> bool:
        """Set the priority of one of this app's jobs still waiting in the gateway's queue, past its
        hourly limits: the owner's override (POST /gateway/jobs/<id>/priority, which the gateway takes
        only from this PC). False when the job isn't waiting there: already in ComfyUI (running or
        done), or sent to ComfyUI directly, which has no priorities."""
        if not self.gateway_url or base.rstrip("/") != self.gateway_url:
            return False
        try:
            self._request("POST", f"{base}/gateway/jobs/{prompt_id}/priority", json={"priority": priority})
        except ComfyError as e:
            if e.status == 400:  # "not in the gateway's queue"
                return False
            raise
        logger.info("comfy prioritized prompt_id=%s priority=%s", prompt_id, priority)
        return True

    def gateway_job(self, base: str, prompt_id: str) -> dict | None:
        """The gateway's record of one of this app's jobs -- its status (queued, dispatched, running;
        success, error, interrupted, cancelled, rejected, lost), when it was sent, handed to ComfyUI,
        started and finished, and the images it saved -- or None when it went to ComfyUI directly or
        the gateway doesn't know it; GATEWAY_UNKNOWN when it can't be read now."""
        if not self.gateway_url or base.rstrip("/") != self.gateway_url:
            return None
        try:
            record = self._request("GET", f"{base}/gateway/jobs/{prompt_id}", timeout=(CONNECT_TIMEOUT, 10)).json()
        except ComfyError as e:
            # One it doesn't know is gone; any other answer, like no answer, says nothing yet.
            return None if e.status == 404 else dict(GATEWAY_UNKNOWN)
        except (requests.exceptions.RequestException, ValueError):
            return dict(GATEWAY_UNKNOWN)  # restarting, say (5-20 s, on 2026-11-05): ask again next time
        if not isinstance(record, dict):
            return dict(GATEWAY_UNKNOWN)
        return {key: record.get(key)
                for key in ("status", "submitted_at", "dispatched_at", "started_at", "finished_at", "outputs")}

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

    def node_info(self, class_type: str) -> dict | None:
        """ComfyUI's definition of a node type (its /object_info entry: inputs, outputs), or None
        if it has no such node. Raw: it may list files, which the caller filters before the bot
        sees it (chat_tools.describe_node)."""
        cached = self._nodes.get(class_type)
        if cached and time.monotonic() - cached[1] < NODE_INFO_CACHE_SECONDS:
            return cached[0]
        base = self.pick_base()
        response = self._request("GET", f"{base}/object_info/{quote(class_type, safe='')}",
                                 timeout=(CONNECT_TIMEOUT, 30))
        try:
            info = (response.json() or {}).get(class_type)
        except (ValueError, AttributeError):
            info = None
        self._nodes[class_type] = (info, time.monotonic())
        return info

    def models(self, folder: str, max_age: float = MODELS_CACHE_SECONDS, read_timeout: float = 60) -> list[str]:
        if folder not in MODEL_FOLDERS:
            raise ValueError(f"not a model folder: {folder}")
        cached = self._models.get(folder)
        if cached and time.monotonic() - cached[1] < max_age:
            return cached[0]
        base = self.pick_base()
        names = self._request("GET", f"{base}/models/{folder}", timeout=(CONNECT_TIMEOUT, read_timeout)).json()
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

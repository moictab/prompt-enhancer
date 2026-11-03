"""Every request sent to OpenRouter, exactly as sent, with what came back.

One JSONL file per UTC day under data/requests/ (YYYY-MM-DD.jsonl), so old days
can be pruned by deleting files. Each line is one HTTP request: the JSON body
byte-for-byte as posted -- except that each base64 image in it is stored once,
by content hash, under data/requests/images/ and its data URI replaced by
"logged-image:<file>" -- plus the status, timing, and the response: the full
JSON for a normal call, or for a streamed one the text as it arrived with the
final usage. The API key travels in a header and is never logged.

openrouter_client opens a RequestRecord per HTTP request and finishes it with
the outcome; a failure to log is reported but never breaks the call.
"""
import base64
import hashlib
import logging
import re
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from . import storage
from .config import get_settings

logger = logging.getLogger("app.request_log")

# Request ids start with their UTC date and time, which names the day file to
# look in and keeps them in time order.
ID_PATTERN = re.compile(r"^\d{8}-\d{6}-[0-9a-f]{8}$")
IMAGE_NAME_PATTERN = re.compile(r"^[0-9a-f]{64}\.(?:png|jpg|webp|gif|bin)$")
IMAGE_EXTENSIONS = {"image/png": "png", "image/jpeg": "jpg", "image/webp": "webp", "image/gif": "gif"}
IMAGE_CONTENT_TYPES = {ext: content_type for content_type, ext in IMAGE_EXTENSIONS.items()}
IMAGE_MARKER = "logged-image:"
MAX_BODY_CHARS = 5000
PREVIEW_CHARS = 160

_lock = threading.Lock()


def _root(root: str | None) -> str:
    return root or f"{get_settings().data_dir}/requests"


def _day_file(request_id: str, root: str | None) -> str:
    day = request_id[:8]
    return f"{_root(root)}/{day[:4]}-{day[4:6]}-{day[6:8]}.jsonl"


def _store_image(content_type: str, encoded: str, root: str | None) -> str:
    data = base64.b64decode(encoded)
    name = f"{hashlib.sha256(data).hexdigest()}.{IMAGE_EXTENSIONS.get(content_type, 'bin')}"
    path = f"{_root(root)}/images/{name}"
    if not Path(path).exists():
        storage.write_bytes(path, data)
    return name


def _without_images(value, root: str | None):
    if isinstance(value, dict):
        return {key: _without_images(item, root) for key, item in value.items()}
    if isinstance(value, list):
        return [_without_images(item, root) for item in value]
    if isinstance(value, str) and value.startswith("data:image/"):
        header, _, encoded = value.partition(",")
        if header.endswith(";base64"):
            return IMAGE_MARKER + _store_image(header[len("data:"):-len(";base64")], encoded, root)
    return value


class RequestRecord:
    """One HTTP request to OpenRouter. Created just before it is sent; finish()
    writes it to the log, once, with whatever came back."""

    def __init__(self, payload: dict, context: dict | None = None, attempt: int = 1, root: str | None = None):
        self.started = datetime.now(timezone.utc)
        self.id = f"{self.started:%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:8]}"
        self.payload = payload
        self.context = dict(context or {})
        self.attempt = attempt
        self.root = root
        self._clock = time.monotonic()
        self._finished = False

    def finish(
        self,
        status: int | None = None,
        response: dict | None = None,
        error: str | None = None,
        response_body: str | None = None,
    ) -> None:
        if self._finished:
            return
        self._finished = True
        try:
            entry = {
                "id": self.id,
                "timestamp": self.started.isoformat(),
                "mode": self.context.pop("mode", None),
                "context": self.context,
                "model": self.payload.get("model"),
                "stream": bool(self.payload.get("stream")),
                "attempt": self.attempt,
                "duration_ms": round((time.monotonic() - self._clock) * 1000),
                "status": status,
                "error": error,
                "request": _without_images(self.payload, self.root),
                "response": response,
                "response_body": response_body[:MAX_BODY_CHARS] if response_body else None,
            }
            with _lock:
                storage.append_jsonl(_day_file(self.id, self.root), entry)
        except Exception:
            logger.exception("could not log request id=%s", self.id)


def _text_of(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text")
    return ""


def response_text(entry: dict) -> str:
    """The reply's text, whether the call was streamed or not."""
    response = entry.get("response") or {}
    if "content" in response:
        return response.get("content") or ""
    try:
        return response["choices"][0]["message"]["content"] or ""
    except (KeyError, IndexError, TypeError):
        return ""


def summary(entry: dict) -> dict:
    messages = (entry.get("request") or {}).get("messages") or []
    response = entry.get("response") or {}
    usage = response.get("usage") or {}
    last_user = next((m for m in reversed(messages) if m.get("role") == "user"), {})
    preview = " ".join(_text_of(last_user.get("content")).split())
    if len(preview) > PREVIEW_CHARS:
        preview = preview[:PREVIEW_CHARS].rstrip() + "…"
    finish_reason = response.get("finish_reason")
    if finish_reason is None:
        try:
            finish_reason = response["choices"][0].get("finish_reason")
        except (KeyError, IndexError, TypeError, AttributeError):
            finish_reason = None
    return {
        "id": entry["id"],
        "timestamp": entry["timestamp"],
        "mode": entry.get("mode"),
        "context": entry.get("context") or {},
        "model": entry.get("model"),
        "stream": entry.get("stream"),
        "attempt": entry.get("attempt"),
        "duration_ms": entry.get("duration_ms"),
        "status": entry.get("status"),
        "error": entry.get("error"),
        "cost": usage.get("cost"),
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "finish_reason": finish_reason,
        "message_count": len(messages),
        "preview": preview,
    }


def _day_files(root: str | None) -> list[Path]:
    directory = Path(_root(root))
    if not directory.is_dir():
        return []
    return sorted(directory.glob("????-??-??.jsonl"), reverse=True)


def list_summaries(
    limit: int = 50,
    before: str | None = None,
    mode: str | None = None,
    errors_only: bool = False,
    root: str | None = None,
) -> dict:
    """Newest first. `before` is the id of the last request already shown: the page
    continues right after it."""
    items = []
    waiting_for_cursor = bool(before)
    cursor_file = Path(_day_file(before, root)).name if before else None
    for day_file in _day_files(root):
        if cursor_file and day_file.name > cursor_file:
            continue
        for entry in reversed(storage.read_jsonl(str(day_file))):
            if waiting_for_cursor:
                waiting_for_cursor = entry["id"] != before
                continue
            if mode and entry.get("mode") != mode:
                continue
            if errors_only and not entry.get("error"):
                continue
            items.append(summary(entry))
            if len(items) > limit:
                return {"items": items[:limit], "has_more": True}
        # The cursor's day is done: later days start from their newest entry.
        waiting_for_cursor = False
    return {"items": items, "has_more": False}


def get(request_id: str, root: str | None = None) -> dict | None:
    if not ID_PATTERN.match(request_id or ""):
        return None
    for entry in storage.read_jsonl(_day_file(request_id, root)):
        if entry["id"] == request_id:
            return entry
    return None


def image_file(name: str, root: str | None = None) -> tuple[str, str] | None:
    if not IMAGE_NAME_PATTERN.match(name or ""):
        return None
    path = Path(f"{_root(root)}/images/{name}")
    if not path.is_file():
        return None
    return str(path), IMAGE_CONTENT_TYPES.get(name.rsplit(".", 1)[1], "application/octet-stream")


def storage_bytes(root: str | None = None) -> int:
    directory = Path(_root(root))
    if not directory.is_dir():
        return 0
    return sum(p.stat().st_size for p in directory.rglob("*") if p.is_file())

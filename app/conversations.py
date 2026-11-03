"""Conversar's conversations: one JSON file per conversation, plus a folder of
the images shared in it, both named by the conversation id.

    data/conversations/<id>.json
    data/conversations/<id>/<image id>.<ext>

A conversation grows with every message, so it gets its own file rather than
living in one shared list rewritten wholesale on each message.
"""
import re
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

from . import storage
from .config import get_settings

IMAGE_EXTENSIONS = {"image/jpeg": "jpg", "image/png": "png", "image/webp": "webp"}
_CONTENT_TYPES = {ext: content_type for content_type, ext in IMAGE_EXTENSIONS.items()}
# Ids end up in file paths, so only ids this module could have generated are accepted.
_ID_PATTERN = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")
FALLBACK_TITLE_LENGTH = 60

_lock = threading.Lock()


def _root(root: str | None) -> str:
    return root or f"{get_settings().data_dir}/conversations"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def is_valid_id(value: str) -> bool:
    return bool(_ID_PATTERN.match(value or ""))


def _path(conversation_id: str, root: str | None) -> str:
    return f"{_root(root)}/{conversation_id}.json"


def create(
    family_id: str,
    llm_model: str,
    temperature: float,
    character_ids: list[str],
    root: str | None = None,
) -> dict:
    now = _now()
    conversation = {
        "id": str(uuid.uuid4()),
        "title": "",
        "created_at": now,
        "updated_at": now,
        "family_id": family_id,
        "llm_model": llm_model,
        "temperature": temperature,
        "character_ids": character_ids,
        "cost": None,
        "messages": [],
    }
    storage.write_json(_path(conversation["id"], root), conversation)
    return conversation


def get(conversation_id: str, root: str | None = None) -> dict | None:
    if not is_valid_id(conversation_id):
        return None
    return storage.read_json(_path(conversation_id, root), default=None)


def display_title(conversation: dict) -> str:
    """The bot-given title, or the start of the first message until there is one."""
    if conversation.get("title"):
        return conversation["title"]
    first = next((m for m in conversation["messages"] if m["role"] == "user"), None)
    text = " ".join((first or {}).get("content", "").split())
    if len(text) > FALLBACK_TITLE_LENGTH:
        text = text[:FALLBACK_TITLE_LENGTH].rstrip() + "…"
    return text


def summary(conversation: dict) -> dict:
    return {
        "id": conversation["id"],
        "title": display_title(conversation),
        "created_at": conversation["created_at"],
        "updated_at": conversation["updated_at"],
        "message_count": len(conversation["messages"]),
        "cost": conversation["cost"],
        "family_id": conversation["family_id"],
        "llm_model": conversation["llm_model"],
    }


def list_summaries(root: str | None = None) -> list[dict]:
    directory = Path(_root(root))
    if not directory.is_dir():
        return []
    conversations = [
        storage.read_json(str(p), default=None)
        for p in directory.glob("*.json")
        if is_valid_id(p.stem)
    ]
    summaries = [summary(c) for c in conversations if c]
    return sorted(summaries, key=lambda s: s["updated_at"], reverse=True)


def update_settings(conversation_id: str, changes: dict, root: str | None = None) -> dict | None:
    allowed = {"family_id", "llm_model", "temperature", "character_ids"}
    with _lock:
        conversation = get(conversation_id, root)
        if conversation is None:
            return None
        conversation.update({k: v for k, v in changes.items() if k in allowed})
        storage.write_json(_path(conversation_id, root), conversation)
        return conversation


def append_message(
    conversation_id: str, message: dict, title: str | None = None, root: str | None = None
) -> dict | None:
    """Append a message (stamping it with an id and timestamp), add its cost to
    the conversation's, and set the title if the conversation has none yet."""
    with _lock:
        conversation = get(conversation_id, root)
        if conversation is None:
            return None
        now = _now()
        conversation["messages"].append({"id": str(uuid.uuid4()), "timestamp": now, **message})
        if message.get("cost") is not None:
            conversation["cost"] = (conversation["cost"] or 0) + message["cost"]
        if title and not conversation["title"]:
            conversation["title"] = title
        conversation["updated_at"] = now
        storage.write_json(_path(conversation_id, root), conversation)
        return conversation


def delete(conversation_id: str, root: str | None = None) -> bool:
    with _lock:
        if get(conversation_id, root) is None:
            return False
        storage.remove(f"{_root(root)}/{conversation_id}")
        storage.remove(_path(conversation_id, root))
        return True


def save_image(conversation_id: str, data: bytes, content_type: str, root: str | None = None) -> dict:
    image = {"id": uuid.uuid4().hex, "content_type": content_type}
    extension = IMAGE_EXTENSIONS[content_type]
    storage.write_bytes(f"{_root(root)}/{conversation_id}/{image['id']}.{extension}", data)
    return image


def image_file(conversation_id: str, image_id: str, root: str | None = None) -> tuple[str, str] | None:
    """The path and content type of an image shared in a conversation, if it exists."""
    if not is_valid_id(conversation_id) or not re.fullmatch(r"[0-9a-f]{32}", image_id or ""):
        return None
    for extension, content_type in _CONTENT_TYPES.items():
        path = Path(f"{_root(root)}/{conversation_id}/{image_id}.{extension}")
        if path.is_file():
            return str(path), content_type
    return None

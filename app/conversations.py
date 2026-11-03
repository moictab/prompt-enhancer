"""Conversar's conversations: one JSON file per conversation, plus a folder of
the images shared in it, both named by the conversation id.

    data/conversations/<id>.json
    data/conversations/<id>/<image id>.<ext>            the image as shared or generated
    data/conversations/<id>/<image id>-small.jpg        a reduced copy, for large images
    data/conversations/<id>/<image id>.workflow.json    the ComfyUI workflow it embeds
    data/conversations/<id>/run-<n>.json                the workflows sent for run n

A conversation grows with every message, so it gets its own file rather than
living in one shared list rewritten wholesale on each message. Messages are
"user", "assistant", or "results" (a finished run's images, added by runs.py).
"""
import re
import threading
import unicodedata
import uuid
from datetime import datetime, timezone
from pathlib import Path

from . import image_metadata, storage
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
        if message.get("role") == "user":
            # The user's message carries their reactions since the last one.
            reactions = _mark_reactions_sent(conversation)
            if reactions:
                message = {**message, "reactions": reactions}
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


def save_image(
    conversation_id: str,
    data: bytes,
    content_type: str,
    root: str | None = None,
    workflow: dict | None = None,
    metadata: dict | None = None,
) -> dict:
    """Store an image, plus a smaller JPEG copy when it is large (what the LLM and the
    thumbnails get) and the ComfyUI workflow it embeds, if any. A run's output passes the
    workflow it was made with instead: its metadata is the run itself. An image shared from
    Civitai passes the metadata Civitai gives (and its workflow, if any)."""
    image = {"id": uuid.uuid4().hex, "content_type": content_type}
    folder = f"{_root(root)}/{conversation_id}"
    storage.write_bytes(f"{folder}/{image['id']}.{IMAGE_EXTENSIONS[content_type]}", data)
    small = image_metadata.small_copy(data)
    if small:
        storage.write_bytes(f"{folder}/{image['id']}-small.jpg", small)
        image["small"] = True
    if metadata is None and workflow is None:
        metadata, workflow = image_metadata.read(data)
    if metadata:
        image["metadata"] = metadata
    if workflow:
        storage.write_json(f"{folder}/{image['id']}.workflow.json", workflow)
    return image


def _valid_image_id(conversation_id: str, image_id: str) -> bool:
    return is_valid_id(conversation_id) and bool(re.fullmatch(r"[0-9a-f]{32}", image_id or ""))


def image_file(
    conversation_id: str, image_id: str, root: str | None = None, small: bool = False
) -> tuple[str, str] | None:
    """The path and content type of an image shared in a conversation, if it exists;
    with small, its reduced copy when it has one."""
    if not _valid_image_id(conversation_id, image_id):
        return None
    if small:
        path = Path(f"{_root(root)}/{conversation_id}/{image_id}-small.jpg")
        if path.is_file():
            return str(path), "image/jpeg"
    for extension, content_type in _CONTENT_TYPES.items():
        path = Path(f"{_root(root)}/{conversation_id}/{image_id}.{extension}")
        if path.is_file():
            return str(path), content_type
    return None


def image_workflow(conversation_id: str, image_id: str, root: str | None = None) -> dict | None:
    """The ComfyUI workflow an image embeds (or a run's output was made with)."""
    if not _valid_image_id(conversation_id, image_id):
        return None
    return storage.read_json(f"{_root(root)}/{conversation_id}/{image_id}.workflow.json", default=None)


# --- reactions: the user's verdict on an image, sent with their next message ------------------

MAX_REACTION_COMMENT = 500


def set_reaction(
    conversation_id: str, image_id: str, rating: str | None, comment: str, root: str | None = None
) -> dict | None:
    """Set (or, with no rating and no comment, clear) the reaction to an image. It stays
    unsent until the user's next message carries it to the bot. Returns the image."""
    with _lock:
        conversation = get(conversation_id, root)
        if conversation is None:
            return None
        image = next((img for m in conversation["messages"] for img in m.get("images") or []
                      if img["id"] == image_id), None)
        if image is None:
            return None
        comment = " ".join((comment or "").split())[:MAX_REACTION_COMMENT]
        if rating is None and not comment:
            image.pop("reaction", None)
        else:
            image["reaction"] = {"rating": rating, "comment": comment, "at": _now(), "sent": False}
        storage.write_json(_path(conversation_id, root), conversation)
        return image


def unsent_reactions(conversation: dict) -> list[dict]:
    return [
        {"image_id": img["id"], "rating": img["reaction"].get("rating"), "comment": img["reaction"].get("comment", "")}
        for m in conversation["messages"] for img in m.get("images") or []
        if img.get("reaction") and not img["reaction"].get("sent")
    ]


def _mark_reactions_sent(conversation: dict) -> list[dict]:
    taken = unsent_reactions(conversation)
    for m in conversation["messages"]:
        for img in m.get("images") or []:
            if img.get("reaction"):
                img["reaction"]["sent"] = True
    return taken


# --- runs: workflows launched on ComfyUI from the conversation ---------------------------------

OUTPUT_SLUG_LENGTH = 40


def slugify(text: str, max_length: int = OUTPUT_SLUG_LENGTH) -> str:
    """Lowercase ASCII words joined by hyphens: safe as a folder name on any system."""
    text = unicodedata.normalize("NFKD", text or "").encode("ascii", "ignore").decode("ascii").lower()
    text = re.sub(r"[^a-z0-9]+", "-", text).strip("-")
    return text[:max_length].rstrip("-")


def output_folder(conversation_id: str, title_hint: str = "", root: str | None = None) -> str | None:
    """The folder, under ComfyUI's output/prompt-enhancer/, that this conversation's images
    are saved in: its creation day, its title (or the one the bot is giving it right now, or
    the start of the first message) and the start of its id. Named at the first run and
    kept, so every run of the conversation lands in the same place."""
    with _lock:
        conversation = get(conversation_id, root)
        if conversation is None:
            return None
        if conversation.get("output_folder"):
            return conversation["output_folder"]
        first = next((m["content"] for m in conversation["messages"]
                      if m["role"] == "user" and m.get("content", "").strip()), "")
        name = slugify(conversation.get("title") or title_hint or " ".join(first.split()[:6])) or "sin-titulo"
        conversation["output_folder"] = f"{conversation['created_at'][:10]}_{name}_{conversation['id'][:8]}"
        storage.write_json(_path(conversation_id, root), conversation)
        return conversation["output_folder"]


def add_run(conversation_id: str, run: dict, workflows: list[dict], root: str | None = None) -> dict | None:
    """Record a run (numbered from 1 within the conversation) and the exact workflow
    sent for each of its jobs."""
    with _lock:
        conversation = get(conversation_id, root)
        if conversation is None:
            return None
        runs = conversation.setdefault("runs", [])
        run = {**run, "n": len(runs) + 1, "created_at": _now()}
        runs.append(run)
        storage.write_json(f"{_root(root)}/{conversation_id}/run-{run['n']}.json", workflows)
        storage.write_json(_path(conversation_id, root), conversation)
        return run


def get_run(conversation_id: str, n: int, root: str | None = None) -> dict | None:
    conversation = get(conversation_id, root)
    runs = (conversation or {}).get("runs") or []
    return runs[n - 1] if 0 < n <= len(runs) else None


def update_run(conversation_id: str, n: int, change, root: str | None = None) -> dict | None:
    """Apply change(run) to the stored run under the lock; returns the run."""
    with _lock:
        conversation = get(conversation_id, root)
        runs = (conversation or {}).get("runs") or []
        if not 0 < n <= len(runs):
            return None
        change(runs[n - 1])
        storage.write_json(_path(conversation_id, root), conversation)
        return runs[n - 1]


def run_workflows(conversation_id: str, n: int, root: str | None = None) -> list[dict]:
    if not is_valid_id(conversation_id):
        return []
    return storage.read_json(f"{_root(root)}/{conversation_id}/run-{int(n)}.json", default=[])


def set_results_images(conversation_id: str, n: int, images: list[dict], root: str | None = None) -> bool:
    """Add images to run n's results message, if it has one (a retried download):
    whether there was one."""
    with _lock:
        conversation = get(conversation_id, root)
        message = next((m for m in (conversation or {}).get("messages") or []
                        if m["role"] == "results" and m.get("run") == n), None)
        if message is None:
            return False
        message["images"] = (message.get("images") or []) + images
        conversation["updated_at"] = _now()
        storage.write_json(_path(conversation_id, root), conversation)
        return True


def pending_runs(root: str | None = None) -> list[tuple[str, dict]]:
    """(conversation id, run) for every run still waiting for ComfyUI, in any conversation."""
    directory = Path(_root(root))
    if not directory.is_dir():
        return []
    found = []
    for p in directory.glob("*.json"):
        if not is_valid_id(p.stem):
            continue
        conversation = storage.read_json(str(p), default=None) or {}
        found += [(p.stem, run) for run in conversation.get("runs") or [] if run.get("status") == "pending"]
    return found

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
# Images the bot may make on its own after each of the user's messages, by default.
INITIATIVE_IMAGES = 24
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
    root: str | None = None,
    auto_comment: bool = True,
    engine: str = "openrouter",
    claude_model: str = "",
    initiative_images: int | None = None,
    claude_effort: str = "",
    claude_auto_model: str = "",
    image_check: bool = True,
) -> dict:
    """auto_comment: when a run finishes, the bot looks at its images on its own; off, they
    wait for the user's next message (one request fewer per run). Older conversations,
    without the field, comment (comments_on_results). engine: who answers, the LLM on
    OpenRouter (llm_model) or Claude Code (claude_model, and claude_effort: "" for the model's own;
    claude_code.py). initiative_images: how
    many images the bot may make on its own after each of the user's messages (initiative)."""
    now = _now()
    conversation = {
        "id": str(uuid.uuid4()),
        "title": "",
        "created_at": now,
        "updated_at": now,
        "family_id": family_id,
        "llm_model": llm_model,
        "temperature": temperature,
        "auto_comment": auto_comment,
        "engine": engine,
        "claude_model": claude_model,
        "claude_effort": claude_effort,
        # What the bot does on its own (its look at a run's images, carrying on alone) on another,
        # cheaper Claude model; "" for the same one.
        "claude_auto_model": claude_auto_model,
        # On Claude Code, a separate inspection of each run's images before the bot sees them (image_check.py).
        "image_check": image_check,
        "initiative_images": INITIATIVE_IMAGES if initiative_images is None else initiative_images,
        "cost": None,
        "messages": [],
    }
    storage.write_json(_path(conversation["id"], root), conversation)
    return conversation


def initiative(conversation: dict) -> int:
    """How many images the bot may make on its own after each of the user's messages (0: only
    what the user asks for). Conversations from before the setting get the default."""
    return conversation.get("initiative_images", INITIATIVE_IMAGES)


def comments_on_results(conversation: dict) -> bool:
    """Whether the bot comments on a run's images when they arrive: the user's setting, or
    always while it works on its own (it has to look at them to go on)."""
    return conversation.get("auto_comment", True) or autonomy(conversation) is not None


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
    replies = [m for m in conversation["messages"] if m["role"] in ("assistant", "results")]
    return {
        "id": conversation["id"],
        "title": display_title(conversation),
        "created_at": conversation["created_at"],
        "updated_at": conversation["updated_at"],
        "message_count": len(conversation["messages"]),
        "cost": conversation["cost"],
        "family_id": conversation["family_id"],
        "llm_model": conversation["llm_model"],
        # When the bot (or a run) last added something: the chat marks what arrived unseen.
        "last_reply_at": replies[-1].get("timestamp") if replies else None,
        "running": any(r["status"] == "pending" for r in conversation.get("runs") or []),
        # While the bot works on its own there: how much of its budget it has used.
        "autonomy": ({k: working.get(k) for k in ("kind", "images", "max_images", "cost", "max_cost")}
                     if (working := autonomy(conversation)) else None),
        # The user ended it (Terminar: the bot's balance, when runs made images); writing reopens it.
        "finished_at": conversation.get("finished_at"),
        # Claude Code's limit was reached: the last message is answered again then (retry_later).
        "retry_at": conversation.get("retry_at"),
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
    """Change settings, or the title (the user's: the bot only titles an untitled
    conversation). Neither counts as activity: updated_at, the list's order, stays."""
    allowed = {"family_id", "llm_model", "temperature", "title", "auto_comment", "engine", "claude_model", "claude_effort", "claude_auto_model",
               "initiative_images", "image_check"}
    with _lock:
        conversation = get(conversation_id, root)
        if conversation is None:
            return None
        conversation.update({k: v for k, v in changes.items() if k in allowed})
        storage.write_json(_path(conversation_id, root), conversation)
        return conversation


def finish(conversation_id: str, root: str | None = None) -> dict | None:
    """End a conversation that has nothing to take stock of (no run made images): it is only marked."""
    with _lock:
        conversation = get(conversation_id, root)
        if conversation is None:
            return None
        conversation["finished_at"] = _now()
        storage.write_json(_path(conversation_id, root), conversation)
        return conversation


def set_claude_code(conversation_id: str, state: dict | None, root: str | None = None) -> dict | None:
    """Claude Code's session for the conversation (claude_code.py): its id, what its recorded
    system prompt held, what context was sent last; None forgets it (the next turn starts a new one)."""
    with _lock:
        conversation = get(conversation_id, root)
        if conversation is None:
            return None
        if state is None:
            conversation.pop("claude_code", None)
        else:
            conversation["claude_code"] = state
        storage.write_json(_path(conversation_id, root), conversation)
        return conversation


def set_topics(conversation_id: str, topics: list[str], root: str | None = None) -> dict | None:
    """What the conversation is about, as the bot declares it (<topics>): its topics' notes come
    with every turn (memory.notes_for_turn)."""
    with _lock:
        conversation = get(conversation_id, root)
        if conversation is None:
            return None
        if topics:
            conversation["topics"] = topics
        else:
            conversation.pop("topics", None)
        storage.write_json(_path(conversation_id, root), conversation)
        return conversation


def set_retry(conversation_id: str, at: int | None, root: str | None = None) -> dict | None:
    """When to answer the last message again by itself (retry_later.py: Claude Code's limit was
    reached, epoch seconds), or None for never. A reply, or a message from the user, clears it."""
    with _lock:
        conversation = get(conversation_id, root)
        if conversation is None:
            return None
        if at is None:
            conversation.pop("retry_at", None)
        else:
            conversation["retry_at"] = at
        storage.write_json(_path(conversation_id, root), conversation)
        return conversation


def discard_last_reply(conversation_id: str, root: str | None = None) -> tuple[dict, dict] | None:
    """Take the bot's last reply off the conversation (to have it answer again: "Rehacer"),
    marking the message it answered as answered again (redo: how many times). Returns the
    reply taken and the conversation, or None when the last message isn't a reply. Its cost stays
    in the conversation's: it was spent."""
    with _lock:
        conversation = get(conversation_id, root)
        if conversation is None or not conversation["messages"] or conversation["messages"][-1]["role"] != "assistant":
            return None
        reply = conversation["messages"].pop()
        if conversation["messages"]:
            answered = conversation["messages"][-1]
            answered["redo"] = answered.get("redo", 0) + 1
        conversation["updated_at"] = _now()
        storage.write_json(_path(conversation_id, root), conversation)
        return reply, conversation


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
            # Asking for the balance ends the conversation; writing again reopens it.
            if message.get("balance"):
                conversation["finished_at"] = now
            else:
                conversation.pop("finished_at", None)
        if message.get("role") in ("user", "assistant"):
            conversation.pop("retry_at", None)  # answered, or the user's next message will be
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


def set_review(conversation_id: str, image_id: str, verdict: str, note: str, root: str | None = None) -> dict | None:
    """The bot's own review of an image a run made: "ok" or "flawed", and what's wrong and why."""
    with _lock:
        conversation = get(conversation_id, root)
        image = next((img for m in (conversation or {}).get("messages") or [] for img in m.get("images") or []
                      if img["id"] == image_id), None)
        if image is None:
            return None
        image["review"] = {"verdict": verdict, "note": " ".join((note or "").split())[:MAX_REACTION_COMMENT],
                           "at": _now()}
        storage.write_json(_path(conversation_id, root), conversation)
        return image


def set_image_checks(conversation_id: str, checks: dict[str, dict], root: str | None = None) -> None:
    """The independent inspection of run images (image_check.py), by image id."""
    with _lock:
        conversation = get(conversation_id, root)
        if conversation is None:
            return
        for message in conversation["messages"]:
            for image in message.get("images") or []:
                if image["id"] in checks:
                    image["check"] = checks[image["id"]]
        storage.write_json(_path(conversation_id, root), conversation)


# Why an image was disliked, picked with a tap (the chat's chips; a comment can add to them).
REACTION_REASONS = ("hands", "face", "anatomy", "style", "composition", "not_asked")
# The same, as the bot reads them.
REASON_WORDS = {"hands": "hands", "face": "face", "anatomy": "anatomy", "style": "style",
                "composition": "composition", "not_asked": "not what was asked"}


def reasons_text(reasons: list[str] | None) -> str:
    return ", ".join(REASON_WORDS.get(r, r) for r in reasons or [])


def set_reaction(
    conversation_id: str, image_id: str, rating: str | None, comment: str, root: str | None = None,
    reasons: list[str] | None = None,
) -> dict | None:
    """Set (or, with no rating and no comment, clear) the reaction to an image: "up", "love" (a
    favorite: the strongest like) or "down" (with its reasons and/or a comment). It stays
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
            picked = [r for r in dict.fromkeys(reasons or []) if r in REACTION_REASONS]
            if rating == "down" and picked:
                image["reaction"]["reasons"] = picked
        storage.write_json(_path(conversation_id, root), conversation)
        return image


def set_inferred_reaction(conversation_id: str, image_id: str, rating: str, comment: str,
                          root: str | None = None) -> dict | None:
    """The bot's reading of the user's words as their reaction to a run's image ("they're all
    great"): stored like a reaction, marked inferred and already sent (the bot knows). One the
    user set themselves stands, and only a run's images take one. Returns {previous} when set
    (the reaction it replaced, if any, to undo it), None otherwise."""
    with _lock:
        conversation = get(conversation_id, root)
        if conversation is None:
            return None
        image = next((img for m in conversation["messages"] if m["role"] == "results"
                      for img in m.get("images") or [] if img["id"] == image_id), None)
        if image is None:
            return None
        previous = image.get("reaction")
        if previous and not previous.get("inferred"):
            return None
        comment = " ".join((comment or "").split())[:MAX_REACTION_COMMENT]
        image["reaction"] = {"rating": rating, "comment": comment, "at": _now(), "sent": True, "inferred": True}
        storage.write_json(_path(conversation_id, root), conversation)
        return {"previous": previous}


def unsent_reactions(conversation: dict) -> list[dict]:
    return [
        {"image_id": img["id"], "rating": img["reaction"].get("rating"), "comment": img["reaction"].get("comment", ""),
         **({"reasons": img["reaction"]["reasons"]} if img["reaction"].get("reasons") else {})}
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


def add_download(conversation_id: str, record: dict, root: str | None = None) -> dict | None:
    """A download from Civitai the bot prepared (downloads.py), numbered from 1 within the conversation."""
    with _lock:
        conversation = get(conversation_id, root)
        if conversation is None:
            return None
        found = conversation.setdefault("downloads", [])
        record = {**record, "n": len(found) + 1}
        found.append(record)
        storage.write_json(_path(conversation_id, root), conversation)
        return record


def get_download(conversation_id: str, n: int, root: str | None = None) -> dict | None:
    found = (get(conversation_id, root) or {}).get("downloads") or []
    return found[n - 1] if isinstance(n, int) and 0 < n <= len(found) else None


def update_download(conversation_id: str, n: int, change, root: str | None = None) -> dict | None:
    """Apply change(record) to the stored download under the lock; returns the record."""
    with _lock:
        conversation = get(conversation_id, root)
        found = (conversation or {}).get("downloads") or []
        if not 0 < n <= len(found):
            return None
        change(found[n - 1])
        storage.write_json(_path(conversation_id, root), conversation)
        return found[n - 1]


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


# --- autonomy: the bot carrying on alone toward a goal, within a budget --------------------------

def autonomy(conversation: dict) -> dict | None:
    """The bot's autonomy while it works on its own there, or None."""
    found = conversation.get("autonomy")
    return found if found and found.get("active") else None


def start_initiative(conversation_id: str, max_images: int, max_cost: float,
                     root: str | None = None) -> dict | None:
    """The user wrote: the bot may carry on alone, until their next message, within max_images
    images and max_cost dollars (a fresh budget, kind "initiative"). A task it was given with
    keep_going (kind "goal") is left as it is: the user's message only steers it."""
    with _lock:
        conversation = get(conversation_id, root)
        if conversation is None:
            return None
        current = autonomy(conversation)
        if current is not None and current.get("kind") != "initiative":
            return current
        conversation["autonomy"] = {
            "active": True, "kind": "initiative", "goal": "", "max_images": max_images, "max_cost": max_cost,
            "images": 0, "cost": 0.0, "last_turn_cost": 0.0, "started_at": _now(),
        }
        storage.write_json(_path(conversation_id, root), conversation)
        return conversation["autonomy"]


def start_autonomy(conversation_id: str, goal: str, max_images: int, max_cost: float,
                   root: str | None = None) -> dict | None:
    """The bot works on its own toward goal (keep_going, kind "goal"): it launches, looks and
    launches again, until the goal is met or it has made max_images images or spent max_cost
    dollars on its replies. Starting again (a new goal or new limits) keeps what was used so far,
    as does turning the budget between messages into a goal."""
    with _lock:
        conversation = get(conversation_id, root)
        if conversation is None:
            return None
        current = autonomy(conversation) or {}
        conversation["autonomy"] = {
            "active": True,
            "kind": "goal",
            "goal": goal,
            "max_images": max_images,
            "max_cost": max_cost,
            "images": current.get("images", 0),
            "cost": current.get("cost", 0.0),
            "last_turn_cost": current.get("last_turn_cost", 0.0),
            "started_at": current.get("started_at") or _now(),
        }
        storage.write_json(_path(conversation_id, root), conversation)
        return conversation["autonomy"]


def add_autonomy_usage(conversation_id: str, images: int = 0, turn_cost: float | None = None,
                       root: str | None = None) -> dict | None:
    """Count images launched, or a reply's cost, against the budget (while the bot works alone)."""
    with _lock:
        conversation = get(conversation_id, root)
        current = autonomy(conversation or {})
        if current is None:
            return None
        current["images"] += images
        if turn_cost is not None:
            current["cost"] = round(current["cost"] + turn_cost, 6)
            current["last_turn_cost"] = turn_cost
        storage.write_json(_path(conversation_id, root), conversation)
        return current


def end_autonomy(conversation_id: str, reason: str, root: str | None = None) -> dict | None:
    """reason: "done" (the bot stopped: goal met, or no better idea), "images" or "cost" (the
    budget ran out), "user" (the user stopped it, or cancelled a run)."""
    with _lock:
        conversation = get(conversation_id, root)
        current = autonomy(conversation or {})
        if current is None:
            return None
        current.update(active=False, ended_at=_now(), end_reason=reason)
        storage.write_json(_path(conversation_id, root), conversation)
        return current

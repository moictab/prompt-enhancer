"""What the Conversar bot knows about the models installed in ComfyUI.

data/catalog.json holds, per file (checkpoints, diffusion models, LoRAs), what the files
themselves say -- for LoRAs, the base model and trigger words from the .safetensors
metadata (ComfyUI's /view_metadata) -- plus the owner's own description of what each one
does, written in Ajustes. Refreshed from ComfyUI on demand and once a day in the
background; descriptions survive a refresh (and a file that disappears for a while).

On top of that, how each file has done here: how many images the bot's runs made with
it and how many of those the user liked or disliked (reactions in Conversar). That,
the descriptions and the names go into the bot's system prompt, compactly.
"""
import json
import logging
import re
import threading
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from . import storage
from .config import get_settings

logger = logging.getLogger("app.catalog")

FOLDERS = {"checkpoints": "Checkpoints", "diffusion_models": "Diffusion models", "loras": "LoRAs"}
# 3D and video models live in the same folders; they don't make images.
NOT_IMAGE = re.compile(r"hunyuan_3d|trellis|pixal3d|wan_2|sam3d|tripo", re.IGNORECASE)
REFRESH_AFTER_SECONDS = 24 * 3600
STATS_CACHE_SECONDS = 30
MAX_TRIGGERS = 6
MAX_DESCRIPTION = 200

_refreshing = threading.Lock()
_stats_cache: dict = {"at": 0.0, "stats": None}


def _default_path() -> str:
    return f"{get_settings().data_dir}/catalog.json"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load(path: str | None = None) -> dict:
    data = storage.read_json(path or _default_path(), default=None) or {}
    return {"refreshed_at": data.get("refreshed_at"), "entries": data.get("entries") or {},
            "descriptions": data.get("descriptions") or {}}


def _save(data: dict, path: str | None = None) -> None:
    storage.write_json(path or _default_path(), data)


def short_name(file: str) -> str:
    return re.sub(r"\.safetensors$", "", file, flags=re.IGNORECASE)


def family_of(file: str) -> str:
    """The folder a file sits in within its models folder ("krea-2\\x" -> "krea-2"), or ""."""
    parts = re.split(r"[\\/]", file)
    return parts[0] if len(parts) > 1 else ""


def _triggers(metadata: dict) -> list[str]:
    phrase = metadata.get("modelspec.trigger_phrase")
    if isinstance(phrase, str) and phrase.strip():
        return [t.strip() for t in phrase.split(",") if t.strip()][:MAX_TRIGGERS]
    try:
        frequency = json.loads(metadata.get("ss_tag_frequency") or "{}")
    except (TypeError, ValueError):
        return []
    counts: Counter = Counter()
    for tags in frequency.values() if isinstance(frequency, dict) else []:
        if isinstance(tags, dict):
            counts.update({t.strip(): n for t, n in tags.items() if isinstance(n, int) and t.strip()})
    return [tag for tag, _ in counts.most_common(MAX_TRIGGERS)]


def entry_from_metadata(file: str, folder: str, metadata: dict | None) -> dict:
    entry = {"folder": folder, "family": family_of(file)}
    metadata = metadata if isinstance(metadata, dict) else {}
    base = metadata.get("ss_base_model_version") or metadata.get("modelspec.architecture")
    if isinstance(base, str) and base.strip():
        entry["base_model"] = base.strip().removesuffix("/lora")
    triggers = _triggers(metadata)
    if triggers:
        entry["triggers"] = triggers
    for key, name in (("modelspec.title", "title"), ("modelspec.description", "about")):
        value = metadata.get(key)
        if isinstance(value, str) and value.strip():
            entry[name] = value.strip()[:MAX_DESCRIPTION]
    return entry


def refresh(client, path: str | None = None) -> dict:
    """Re-read the installed files (and LoRA metadata) from ComfyUI."""
    with _refreshing:
        data = load(path)
        old = data["entries"]
        entries = {}
        for folder in FOLDERS:
            for file in client.models(folder):
                if NOT_IMAGE.search(file):
                    continue
                if folder == "loras":
                    metadata = None
                    if file in old and old[file].get("folder") == folder and "checked" in old[file]:
                        entries[file] = old[file]
                        continue
                    checked = True
                    try:
                        metadata = client.metadata(folder, file)
                    except Exception as e:
                        # A busy ComfyUI can stall for a while: try this one again next time.
                        logger.info("catalog metadata unavailable file=%s error=%r", file, e)
                        checked = False
                    entries[file] = entry_from_metadata(file, folder, metadata)
                    if checked:
                        entries[file]["checked"] = True
                else:
                    entries[file] = entry_from_metadata(file, folder, None)
        data.update(entries=entries, refreshed_at=_now())
        _save(data, path)
        logger.info("catalog refreshed files=%d", len(entries))
        return data


def refresh_in_background(client, force: bool = False) -> bool:
    """Start a refresh unless one is running or the catalog is recent; whether one started."""
    data = load()
    if not force and data["refreshed_at"]:
        age = (datetime.now(timezone.utc) - datetime.fromisoformat(data["refreshed_at"])).total_seconds()
        if age < REFRESH_AFTER_SECONDS:
            return False
    if _refreshing.locked():
        return False
    path = _default_path()  # resolved now: the thread may outlive the settings it started with

    def run():
        try:
            refresh(client, path)
        except Exception as e:
            logger.warning("catalog refresh failed error=%r", e)

    threading.Thread(target=run, name="catalog-refresh", daemon=True).start()
    return True


def set_description(file: str, text: str, path: str | None = None) -> dict:
    data = load(path)
    text = " ".join((text or "").split())
    if text:
        data["descriptions"][file] = text
    else:
        data["descriptions"].pop(file, None)
    _save(data, path)
    return data


def stats() -> dict[str, dict]:
    """Per file: images the bot's runs made with it, and how many the user liked or disliked."""
    if _stats_cache["stats"] is not None and time.monotonic() - _stats_cache["at"] < STATS_CACHE_SECONDS:
        return _stats_cache["stats"]
    totals: dict[str, Counter] = {}
    root = Path(f"{get_settings().data_dir}/conversations")
    for path in root.glob("*.json") if root.is_dir() else []:
        conversation = storage.read_json(str(path), default=None) or {}
        reactions = {img["id"]: (img.get("reaction") or {}).get("rating")
                     for m in conversation.get("messages") or [] for img in m.get("images") or []}
        for run in conversation.get("runs") or []:
            for job in run.get("jobs") or []:
                files = {lora["name"] for lora in job.get("loras") or []} | set(job.get("models") or [])
                for image_id in job.get("images") or []:
                    rating = reactions.get(image_id)
                    for file in files:
                        counter = totals.setdefault(file, Counter())
                        counter["images"] += 1
                        if rating == "up":
                            counter["liked"] += 1
                        elif rating == "down":
                            counter["disliked"] += 1
    result = {file: dict(counter) for file, counter in totals.items()}
    _stats_cache.update(at=time.monotonic(), stats=result)
    return result


def invalidate_stats() -> None:
    _stats_cache["stats"] = None


def details(file: str, data: dict | None = None, file_stats: dict | None = None) -> dict:
    """Everything known about one file, for the bot's tools."""
    data = data or load()
    entry = {k: v for k, v in (data["entries"].get(file) or {}).items() if k not in ("checked", "folder", "family")}
    if data["descriptions"].get(file):
        entry["description"] = data["descriptions"][file]
    record = (file_stats if file_stats is not None else stats()).get(file)
    if record:
        entry["results_here"] = record
    return entry


def _record_text(record: dict | None) -> str:
    if not record:
        return ""
    text = f"{record['images']} image{'s' if record['images'] != 1 else ''}"
    reactions = [f"{record[k]} {k}" for k in ("liked", "disliked") if record.get(k)]
    return f"{text}, {', '.join(reactions)}" if reactions else text


def prompt_section(data: dict | None = None) -> str | None:
    """The installed models for the system prompt: names grouped by folder, with the owner's
    description and how they did here when known. None until the catalog has been read."""
    data = data or load()
    if not data["entries"]:
        return None
    file_stats = stats()
    lines = [
        "What ComfyUI has installed (file names without .safetensors). After a name: the user's description "
        "of what it does, and how images made with it here went (the user's reactions). list_comfy_models gives "
        "details such as trigger words and base model.",
    ]
    for folder, label in FOLDERS.items():
        files = sorted(f for f, e in data["entries"].items() if e.get("folder") == folder)
        if not files:
            continue
        groups: dict[str, list[str]] = {}
        for file in files:
            extras = [x for x in (data["descriptions"].get(file), _record_text(file_stats.get(file))) if x]
            name = short_name(file)
            family = family_of(file)
            if family:
                name = name[len(family) + 1:]
            groups.setdefault(family, []).append(f"{name} ({'; '.join(extras)})" if extras else name)
        if len(groups) == 1 and "" in groups:
            lines.append(f"{label}: {', '.join(groups[''])}")
            continue
        lines.append(f"{label}:")
        for family, names in groups.items():
            lines.append(f"- {family or '(top level)'}: {', '.join(names)}")
    return "\n".join(lines)


def resolve_file(name: str, folder: str, client) -> str:
    """The installed file a name means: exact, or matching despite case, slashes or a missing
    extension. Unknown names come back as given (ComfyUI's validation will report them)."""
    try:
        installed = client.models(folder)
    except Exception:
        return name
    if name in installed:
        return name

    def key(value: str) -> str:
        return short_name(value.replace("/", "\\")).casefold()

    matches = [f for f in installed if key(f) == key(name)]
    if len(matches) == 1:
        return matches[0]
    tail = [f for f in installed if key(f).split("\\")[-1] == key(name).split("\\")[-1]]
    return tail[0] if len(tail) == 1 else name

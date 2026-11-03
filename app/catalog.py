"""What the Conversar bot knows about the models installed in ComfyUI.

data/catalog.json holds, per file (checkpoints, diffusion models, LoRAs), what the files
themselves say -- for LoRAs, the base model and trigger words from the .safetensors
metadata (ComfyUI's /view_metadata) -- plus the owner's own description of what each one
does, written in Ajustes. Refreshed from ComfyUI on demand and once a day in the
background; descriptions survive a refresh (and a file that disappears for a while).

On top of that, how each file has done here: how many images the bot's runs made with
it and how many of those the user liked or disliked (reactions in Conversar); and its
past use outside this app, imported once from comfy-gateway's history as aggregates
(gateway_history.py: jobs, weights, frequent companions -- never prompts or images).
That, the descriptions and the names go into the bot's system prompt, compactly.
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
# The file list is re-read before a turn when older than this (names only: quick), so a model
# installed minutes ago is known; a stalled ComfyUI gets LISTING_READ_TIMEOUT, then the old list.
LISTING_MAX_AGE_SECONDS = 120
LISTING_READ_TIMEOUT = 8
# Words too common in model names to say which file a name means.
GENERIC_WORDS = {"lora", "loras", "lycoris", "locon", "model", "models", "checkpoint", "style", "version",
                 "the", "and", "for", "with", "merge"}
TYPE_FOLDERS = {"lora": ("loras",), "locon": ("loras",), "lycoris": ("loras",), "dora": ("loras",),
                "checkpoint": ("checkpoints", "diffusion_models"), "model": ("checkpoints", "diffusion_models")}
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
    return {"refreshed_at": data.get("refreshed_at"), "listed_at": data.get("listed_at"),
            "entries": data.get("entries") or {},
            "descriptions": data.get("descriptions") or {}, "history": data.get("history") or {},
            "history_source": data.get("history_source")}


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
        data.update(entries=entries, refreshed_at=_now(), listed_at=_now())
        _save(data, path)
        logger.info("catalog refreshed files=%d", len(entries))
        return data


def refresh_files(client, path: str | None = None) -> dict | None:
    """Re-list the installed files now -- names only, quick. Files new since the last full
    refresh come in without metadata (LoRAs marked unchecked, for refresh() to read).
    None if a full refresh is running (it re-lists too)."""
    if not _refreshing.acquire(blocking=False):
        return None
    try:
        data = load(path)
        old = data["entries"]
        entries = {}
        for folder in FOLDERS:
            for file in client.models(folder, max_age=0, read_timeout=LISTING_READ_TIMEOUT):
                if NOT_IMAGE.search(file):
                    continue
                previous = old.get(file)
                entries[file] = previous if previous and previous.get("folder") == folder else \
                    entry_from_metadata(file, folder, None)
        added = len(set(entries) - set(old))
        data.update(entries=entries, listed_at=_now())
        _save(data, path)
        if added:
            logger.info("catalog listing found new files=%d", added)
        return data
    finally:
        _refreshing.release()


def ensure_fresh(client) -> None:
    """Before a turn: re-list the files if the list is a couple of minutes old, and read the
    metadata of new LoRAs in the background. A ComfyUI that doesn't answer leaves the old list."""
    data = load()
    listed_at = data["listed_at"] or data["refreshed_at"]
    if listed_at:
        age = (datetime.now(timezone.utc) - datetime.fromisoformat(listed_at)).total_seconds()
        if age < LISTING_MAX_AGE_SECONDS:
            return
    try:
        data = refresh_files(client)
    except Exception as e:
        logger.info("catalog listing unavailable error=%r", e)
        return
    if data and any(e.get("folder") == "loras" and "checked" not in e for e in data["entries"].values()):
        refresh_in_background(client, force=True)


def _basename(file: str) -> str:
    return re.split(r"[\\/]", file)[-1]


def _normalized(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", short_name(text or "").lower())


def _words(text: str) -> list[str]:
    return [w for w in re.findall(r"[a-z0-9]+", (text or "").lower())
            if len(w) >= 3 and not w.isdigit() and w not in GENERIC_WORDS]


def match_installed(resource: dict, entries: dict) -> str | None:
    """The installed file a resource named elsewhere (a Civitai image) most likely is: the
    same file name once case, separators and folder are ignored, or else the one file whose
    name holds most of the resource's name (preferring its base model's folder). None when
    nothing matches clearly -- files are often renamed, so that isn't proof it's missing."""
    folders = TYPE_FOLDERS.get(str(resource.get("type") or "").lower())
    candidates = [f for f, e in entries.items() if folders is None or e.get("folder") in folders]
    named_file = resource.get("file") or (resource.get("name") if re.search(r"\.(safetensors|ckpt|pt)$",
                                                                            resource.get("name") or "") else None)
    if named_file:
        stem = _normalized(_basename(named_file))
        exact = [f for f in candidates if stem and _normalized(_basename(f)) == stem]
        if len(exact) == 1:
            return exact[0]
    words = _words(resource.get("name") if not named_file or resource.get("file") else _basename(named_file))
    if len(words) < 2:
        return None
    scored = []
    for f in candidates:
        name = _normalized(_basename(f))
        hits = sum(1 for w in words if w in name)
        if hits >= 2 and hits >= 0.6 * len(words):
            scored.append((hits, f))
    if not scored:
        return None
    best = max(hits for hits, _ in scored)
    top = [f for hits, f in scored if hits == best]
    base = _normalized(resource.get("base_model") or "")
    if len(top) > 1 and base:
        top = [f for f in top if _normalized(family_of(f)) == base] or top
    return top[0] if len(top) == 1 else None


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


def set_history(usage: dict, path: str | None = None) -> dict:
    """Replace the past use (gateway_history.aggregate's output): per file, nothing but numbers
    and the names of other files."""
    data = load(path)
    data["history"] = {
        name: {k: v for k, v in entry.items() if k in ("jobs", "weight", "used_with", "last_used")}
        for name, entry in (usage.get("files") or {}).items() if isinstance(name, str) and isinstance(entry, dict)
    }
    data["history_source"] = {"jobs": usage.get("jobs"), "from": usage.get("from"), "to": usage.get("to"),
                              "imported_at": _now()}
    _save(data, path)
    return data


def set_description(file: str, text: str, path: str | None = None) -> dict:
    data = load(path)
    text = " ".join((text or "").split())
    if text:
        data["descriptions"][file] = text
    else:
        data["descriptions"].pop(file, None)
    _save(data, path)
    return data


MAX_REASONS = 3


def stats() -> dict[str, dict]:
    """Per file, from the images the bot's runs made with it: how many; how many the user liked
    or disliked (and the latest reasons); how many the bot's own review found flawed, unless the
    user liked them (and at which weights, and the latest flaws); and, for a LoRA, what its
    latest calibration found at each weight."""
    if _stats_cache["stats"] is not None and time.monotonic() - _stats_cache["at"] < STATS_CACHE_SECONDS:
        return _stats_cache["stats"]
    totals: dict[str, dict] = {}
    calibrations: dict[str, tuple[str, dict]] = {}
    root = Path(f"{get_settings().data_dir}/conversations")
    for path in root.glob("*.json") if root.is_dir() else []:
        conversation = storage.read_json(str(path), default=None) or {}
        images = {img["id"]: img for m in conversation.get("messages") or [] for img in m.get("images") or []}
        for run in conversation.get("runs") or []:
            calibrated = run.get("lora") if run.get("kind") == "calibration" else None
            verdicts = {}
            for job in run.get("jobs") or []:
                strengths = {lora["name"]: lora.get("strength") for lora in job.get("loras") or []}
                for image_id in job.get("images") or []:
                    image = images.get(image_id) or {}
                    reaction, review = image.get("reaction") or {}, image.get("review") or {}
                    rating = reaction.get("rating")
                    flawed = review.get("verdict") == "flawed" and rating != "up"
                    for file in set(strengths) | set(job.get("models") or []):
                        record = totals.setdefault(file, {"images": 0, "liked": 0, "disliked": 0, "flawed": 0,
                                                          "weights": [], "flaws": [], "reasons": []})
                        weight = strengths.get(file)
                        at = f"at {weight:g}: " if isinstance(weight, (int, float)) else ""
                        record["images"] += 1
                        if rating == "up":
                            record["liked"] += 1
                        elif rating == "down":
                            record["disliked"] += 1
                            record["reasons"].append((reaction.get("at") or "", at + (reaction.get("comment") or "")))
                        if flawed:
                            record["flawed"] += 1
                            if isinstance(weight, (int, float)):
                                record["weights"].append(weight)
                            record["flaws"].append((review.get("at") or "", at + (review.get("note") or "")))
                    weight = job.get("calibration_weight")
                    if calibrated and weight is not None and review.get("verdict"):
                        verdict = "flawed" if flawed else "ok"
                        verdicts[f"{weight:g}"] = verdict + (f": {review.get('note')}" if flawed and review.get("note")
                                                             else "")
            if verdicts and run.get("created_at", "") >= calibrations.get(calibrated, ("", {}))[0]:
                calibrations[calibrated] = (run.get("created_at", ""), verdicts)
    result = {}
    for file, record in totals.items():
        out = {k: record[k] for k in ("images", "liked", "disliked", "flawed") if record[k] or k == "images"}
        if record["weights"]:
            out["flawed_weights"] = [min(record["weights"]), max(record["weights"])]
        for key, label in (("flaws", "latest_flaws"), ("reasons", "latest_dislike_reasons")):
            if record[key]:
                out[label] = [text for _, text in sorted(record[key])[-MAX_REASONS:]]
        result[file] = out
    for file, (_, verdicts) in calibrations.items():
        result.setdefault(file, {"images": 0})["calibration"] = verdicts
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
    if data["history"].get(file):
        entry["past_use"] = data["history"][file]
    record = (file_stats if file_stats is not None else stats()).get(file)
    if record:
        entry["results_here"] = record
    return entry


def _past_use_text(past: dict | None) -> str:
    if not past:
        return ""
    text = f"used in {past['jobs']} past job{'s' if past['jobs'] != 1 else ''}"
    weight = past.get("weight")
    if weight:
        low, high = weight["min"], weight["max"]
        text += f" at ~{weight['median']:g}" + (f" ({low:g}-{high:g})" if low != high else "")
    return text


def _calibration_text(verdicts: dict) -> str:
    weights = sorted((float(w), v) for w, v in verdicts.items())
    flawed = [w for w, v in weights if v.startswith("flawed")]
    if not flawed:
        return f"calibrated: fine up to {weights[-1][0]:g}"
    fine = [w for w, v in weights if v == "ok" and w < flawed[0]]
    return (f"calibrated: fine up to {fine[-1]:g}, flawed from {flawed[0]:g}" if fine
            else f"calibrated: flawed from {flawed[0]:g}")


def _record_text(record: dict | None) -> str:
    if not record:
        return ""
    parts = []
    if record.get("images"):
        parts.append(f"{record['images']} image{'s' if record['images'] != 1 else ''}")
        parts += [f"{record[k]} {k}" for k in ("liked", "disliked") if record.get(k)]
        if record.get("flawed"):
            low, high = record.get("flawed_weights") or (None, None)
            where = "" if low is None else (f" at {low:g}" if low == high else f" at {low:g}-{high:g}")
            parts.append(f"{record['flawed']} flawed{where}")
    if record.get("calibration"):
        parts.append(_calibration_text(record["calibration"]))
    return ", ".join(parts)


def prompt_section(data: dict | None = None) -> str | None:
    """The installed models for the system prompt: names grouped by folder, with the owner's
    description and how they did here when known. None until the catalog has been read."""
    data = data or load()
    if not data["entries"]:
        return None
    file_stats = stats()
    lines = [
        "What ComfyUI has installed (file names without .safetensors). After a name: the user's description "
        "of what it does, how much and at what weight the user has used it before, and how images made with it "
        "here went (the user's reactions). list_comfy_models gives details such as trigger words, base model "
        "and the LoRAs it is usually combined with.",
    ]
    for folder, label in FOLDERS.items():
        files = sorted(f for f, e in data["entries"].items() if e.get("folder") == folder)
        if not files:
            continue
        groups: dict[str, list[str]] = {}
        for file in files:
            extras = [x for x in (data["descriptions"].get(file), _past_use_text(data["history"].get(file)),
                                  _record_text(file_stats.get(file))) if x]
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

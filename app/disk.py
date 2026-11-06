"""The models folder against how much each file is used: what can go when space runs short.

Each checkpoint, diffusion model and LoRA in COMFY_MODELS_DIR, with its size, how long it has been on
disk (the older of its creation and modification times: a copy resets the first, not the second) and
its use -- every successful job that loaded it, from comfy-gateway's record (catalog history,
gateway_history.fetch: the app's runs, the other projects' and what was launched by hand in ComfyUI,
which the gateway records too). Use is judged over the time the record covers and the file was there,
so a file just downloaded isn't flagged for being little used yet. Files the library's workflows load
are never flagged: deleting one would break them.

`delete` removes a file the user picked (and its side files: previews, Civitai info) for good.
"""
import logging
import os
import shutil
import time
from datetime import date, datetime
from pathlib import Path

from . import catalog, gateway_history, workflows
from .comfy_graph import Graph, GraphError
from .config import get_settings

logger = logging.getLogger(__name__)

FOLDERS = ("checkpoints", "diffusion_models", "loras")
EXTENSIONS = {".safetensors", ".ckpt", ".pt", ".pth", ".gguf", ".sft", ".bin"}
NEW_DAYS = 14  # on disk for less: too early to judge
STALE_DAYS = 30  # not used for this long: forgotten
RARE_PER_MONTH = 2  # fewer uses than this per 30 days it was there: little used
USAGE_MAX_AGE_SECONDS = 24 * 3600  # the record of use is read again past this


class DiskError(Exception):
    pass


def configured() -> bool:
    return bool(get_settings().comfy_models_dir)


def _root() -> Path:
    folder = get_settings().comfy_models_dir
    if not folder:
        raise DiskError("The models folder isn't set up here (COMFY_MODELS_DIR).")
    return Path(folder)


def _key(name: str) -> str:
    return name.replace("/", "\\").lower()


FRESH_SECONDS = 300  # the report's own reading of the gateway's record serves this long
_fresh: dict = {"at": 0.0, "usage": None}


def usage(force: bool = False) -> tuple[dict, dict]:
    """Each file's use (by name) and where it comes from ({jobs, from, read_at}): the gateway's
    record read for the report itself, at most every FRESH_SECONDS (or forced), so it counts what
    was made minutes ago. The catalog's past use -- what the bot sees -- is replaced from it at
    most once a day (USAGE_MAX_AGE_SECONDS), so the bot's context doesn't change with every look.
    With the gateway unreadable, the catalog's."""
    now = time.time()
    url = get_settings().comfy_gateway_url
    if url and (force or _fresh["usage"] is None or now - _fresh["at"] > FRESH_SECONDS):
        try:
            _fresh.update(usage=gateway_history.fetch(url), at=now)
            logger.info("model usage read from the gateway: %s jobs", _fresh["usage"].get("jobs"))
        except Exception as e:  # the gateway down or restarting: what was read before stands
            logger.warning("model usage not read from the gateway: %s", e)
    stored = catalog.load()
    source = stored["history_source"] or {}
    read = _fresh["usage"]
    if read is None:
        return stored["history"], {"jobs": source.get("jobs"), "from": source.get("from"),
                                   "read_at": source.get("imported_at")}
    imported = source.get("imported_at")
    if not imported or now - datetime.fromisoformat(imported).timestamp() >= USAGE_MAX_AGE_SECONDS:
        catalog.set_history(read)
    return read.get("files") or {}, {"jobs": read.get("jobs"), "from": read.get("from"),
                                     "read_at": datetime.fromtimestamp(_fresh["at"]).astimezone().isoformat()}


def _library() -> dict[str, list[str]]:
    """The library workflows that load each file (by its normalized name)."""
    found: dict[str, list[str]] = {}
    for entry in workflows.list_workflows():
        try:
            graph = Graph(entry["workflow"])
            names = set(graph.models()) | {lora["name"] for lora in graph.loras()}
        except (GraphError, KeyError, TypeError, AttributeError):
            continue
        for name in names:
            found.setdefault(_key(name), []).append(entry["name"])
    return found


def _state(age_days: int, jobs: int, idle_days: int | None, per_month: float, library: list[str]) -> str:
    if library:
        return "library"
    if age_days < NEW_DAYS:
        return "new"
    if not jobs:
        return "unused"
    if idle_days is not None and idle_days >= STALE_DAYS:
        return "stale"
    if per_month < RARE_PER_MONTH:
        return "rare"
    return "used"


CANDIDATES = ("unused", "stale", "rare")


def report(refresh: bool = False, today: date | None = None) -> dict:
    """Every model file with its size, time on disk, use and verdict (_state), biggest first; and
    the disk's free space."""
    root = _root()
    used_files, source = usage(force=refresh)
    history = {_key(name): entry for name, entry in used_files.items()}
    library = _library()
    today = today or date.today()
    records_from = date.fromisoformat(source["from"]) if source.get("from") else today
    files = []
    for folder in FOLDERS:
        base = root / folder
        for path in sorted(base.rglob("*")) if base.is_dir() else []:
            if not path.is_file() or path.suffix.lower() not in EXTENSIONS:
                continue
            info = path.stat()
            since = datetime.fromtimestamp(min(info.st_ctime, info.st_mtime)).date()
            name = str(path.relative_to(base))
            used = history.get(_key(name)) or {}
            jobs = int(used.get("jobs") or 0)
            last = used.get("last_used")
            observed = max(1, (today - max(since, records_from)).days)
            idle = (today - date.fromisoformat(last)).days if last else None
            per_month = jobs * 30 / observed
            refs = library.get(_key(name)) or []
            files.append({
                "folder": folder, "name": name, "bytes": info.st_size, "on_disk_since": since.isoformat(),
                "age_days": (today - since).days, "jobs": jobs, "last_used": last, "observed_days": observed,
                "per_month": round(per_month, 1), "library": refs,
                "state": _state((today - since).days, jobs, idle, per_month, refs),
            })
    files.sort(key=lambda f: -f["bytes"])
    disk = shutil.disk_usage(root)
    return {
        "root": str(root), "free_bytes": disk.free, "total_bytes": disk.total,
        "records_from": source.get("from"), "records_jobs": source.get("jobs"), "usage_read_at": source.get("read_at"),
        "new_days": NEW_DAYS, "stale_days": STALE_DAYS, "rare_per_month": RARE_PER_MONTH,
        "files": files,
        "candidate_bytes": sum(f["bytes"] for f in files if f["state"] in CANDIDATES),
    }


def _model_path(folder: str, name: str) -> Path:
    if folder not in FOLDERS:
        raise DiskError(f"Not a models folder: {folder}")
    base = (_root() / folder).resolve()
    path = (base / name).resolve()
    if base not in path.parents or path.suffix.lower() not in EXTENSIONS:
        raise DiskError("That isn't a model file in the models folder.")
    if not path.is_file():
        raise DiskError("That file isn't there any more.")
    return path


def delete(folder: str, name: str) -> dict:
    """Delete a model file for good, with its side files (same name, another ending: a preview, its
    Civitai info). Refused for a file the library's workflows load. Returns what was removed."""
    path = _model_path(folder, name)
    refs = _library().get(_key(name))
    if refs:
        raise DiskError(f"The library's workflows load it ({', '.join(refs)}): change them first.")
    stem = path.name[: -len(path.suffix)]
    side = [p for p in path.parent.iterdir()
            if p.is_file() and p != path and p.name.startswith(stem + ".") and p.suffix.lower() not in EXTENSIONS]
    size = path.stat().st_size
    removed = []
    for target in [path, *side]:
        os.remove(target)
        removed.append(target.name)
    logger.info("model file deleted folder=%s name=%s bytes=%d side=%d", folder, name, size, len(side))
    return {"removed": removed, "bytes": size}

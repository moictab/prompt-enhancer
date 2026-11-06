"""Downloading models and LoRAs from Civitai into ComfyUI's models folder (COMFY_MODELS_DIR).
The bot prepares each download and only the user starts it.

The bot notices what a conversation needs that isn't installed and Civitai has, for example
a shared Civitai image's resources (their Civitai version), a model link the user pasted, or a
reference image's hash. It prepares a download with the prepare_download tool (`prepare`),
which works out:
- the version and its SafeTensors file;
- where the file would go: the models folder by type (loras, checkpoints, diffusion_models)
  and the family folder by base model ("Illustrious" goes to illustrious);
- the name it would get, in the owner's convention (`file_name`): "char-ramona-flowers_v1.0"
  for a LoRA, "wai_v14.0" for a checkpoint.

The chat shows the prepared download as a card. Nothing is fetched until the user accepts it
there (`accept`), with the folder and name as they left them, and then only:
- from Civitai's own download URL for that version, with the API key sent to civitai.com
  only (requests drops it on the redirect to the file's host);
- a SafeTensors file (pickle formats can run code), checked against Civitai's SHA256 and
  for the format's header;
- into an existing family folder under COMFY_MODELS_DIR, written to a ".part" file and
  renamed when complete, never overwriting a file.

One download runs at a time. Progress is kept in memory (`progress`). What came down is kept
when the connection fails: the download carries on from there (an HTTP Range request), by
itself a few times (`RETRY_WAITS`), and then when the user accepts it again -- also after a
restart cut it short (`after_restart`). Only a cancel, a bad file or a checksum that doesn't
match throws it away. Once installed, the catalog's file list is refreshed, so the next turn
sees the file.
"""
import hashlib
import logging
import re
import shutil
import struct
import threading
import time
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import requests

from . import civitai, comfy, conversations
from .config import get_settings

logger = logging.getLogger(__name__)

TYPE_FOLDERS = {"lora": "loras", "locon": "loras", "lycoris": "loras", "dora": "loras", "checkpoint": "checkpoints"}
MODEL_FOLDERS = ("loras", "checkpoints", "diffusion_models")
DOWNLOAD_URL = "https://civitai.com/api/download/models/"
CHUNK = 1 << 20
SPACE_MARGIN = 1 << 30  # free space left after the file, at least
TIMEOUT = (15, 120)
# Seconds to wait before each new try when the connection fails, carrying on from what came down;
# past the last, the download fails keeping it, for the user to carry on later.
RETRY_WAITS = (5, 15, 30, 60, 120, 120)
MAX_NAME_LENGTH = 120
NAME = re.compile(r"^[a-z0-9][a-z0-9._+-]*$")
# Civitai's categories (among a model's tags), as the owner's LoRA names begin.
CATEGORY_PREFIXES = {
    "character": "char", "celebrity": "char", "style": "style", "concept": "concept", "clothing": "clothing",
    "poses": "action", "action": "action", "base model": "base", "assets": "assets", "objects": "assets",
}
DEFAULT_PREFIX = "concept"
# Words in Civitai names that only say which base model or format it is: the folder says that already.
NOISE_WORDS = {"lora", "locon", "lycoris", "dora", "checkpoint", "model", "illustrious", "ill", "il", "noobai",
               "noob", "pony", "sdxl", "xl", "sd", "sd15", "flux", "krea", "anima", "zit", "zimage", "turbo"}
# Civitai's base models, squashed (letters and digits only), and the family folder they go in when
# no folder of that name exists yet.
BASE_FOLDERS = {"sdxl10": "sdxl", "sdxl": "sdxl", "sdxllightning": "sdxl", "sdxlhyper": "sdxl",
                "illustrious": "illustrious", "pony": "pony", "noobai": "noob-ai", "flux1d": "flux",
                "flux1s": "flux", "flux1dev": "flux", "zimageturbo": "zit", "anima": "anima", "krea2": "krea-2"}

_progress: dict[tuple[str, int], dict] = {}
_cancelled: set[tuple[str, int]] = set()
_lock = threading.Lock()
_one_at_a_time = threading.Semaphore(1)


class DownloadError(ValueError):
    """Something the bot or the user can act on: said as is."""


class _Cancelled(Exception):
    pass


class _Cut(Exception):
    """The file stopped coming before it was whole: try again from where it stopped."""


class _Kept(DownloadError):
    """A failure that keeps what came down: accepting it again carries on from there."""


def models_root() -> Path | None:
    folder = get_settings().comfy_models_dir
    return Path(folder) if folder and Path(folder).is_dir() else None


def family_folders() -> dict[str, list[str]]:
    """The family folders under each models folder: where a download may go."""
    root = models_root()
    if root is None:
        return {}
    return {kind: sorted(p.name for p in (root / kind).iterdir() if p.is_dir() and not p.name.startswith("."))
            for kind in MODEL_FOLDERS if (root / kind).is_dir()}


# --- names ---------------------------------------------------------------------------------------

def _squash(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (text or "").lower())


def _slug(text: str) -> str:
    text = unicodedata.normalize("NFKD", text or "").encode("ascii", "ignore").decode()
    words = [w for w in re.split(r"[^a-z0-9.]+", text.lower()) if w]
    kept = [w.strip(".") for w in words if w.strip(".") and w.strip(".") not in NOISE_WORDS]
    return "-".join(kept or [w.strip(".") for w in words if w.strip(".")])[:60].strip("-")


def _base_name(model_name: str) -> str:
    """The model's own name, without what Civitai titles often add: "Ramona Flowers | Illustrious
    LoRA" -> "ramona-flowers"."""
    head = re.split(r"\s[|/–—-]\s|\s*[\[(:]", model_name or "", maxsplit=1)[0]
    return _slug(head) or _slug(model_name) or "model"


def _version(version_name: str) -> str:
    """"v3" -> "v3.0", "V1.1 Illustrious" -> "v1.1", "Illustrious" -> ""."""
    match = re.search(r"(?<![a-z0-9])v?(\d+(?:\.\d+)*)", (version_name or "").lower())
    if not match:
        return ""
    number = match.group(1)
    return f"v{number if '.' in number else number + '.0'}"


def file_name(model_name: str, version_name: str, kind: str, tags: list[str] | None = None) -> str:
    """The name a file gets here, without extension: "<category>-<name>_v<version>" for a LoRA
    (char, style, concept, clothing, action, base, assets; from Civitai's categories),
    "<name>_v<version>" for a checkpoint."""
    name = _base_name(model_name)
    if kind == "loras":
        lowered = [str(t).lower() for t in tags or []]
        prefix = next((CATEGORY_PREFIXES[t] for t in lowered if t in CATEGORY_PREFIXES), DEFAULT_PREFIX)
        name = f"{prefix}-{name}"
    version = _version(version_name)
    return f"{name}_{version}" if version else name


def _family(base_model: str, kind: str, folders: dict[str, list[str]]) -> str:
    """The family folder for a Civitai base model ("Illustrious" -> "illustrious"), or "" when
    no existing folder fits (the user picks one)."""
    key = _squash(base_model)
    existing = folders.get(kind, [])
    for name in existing:
        if _squash(name) == key or BASE_FOLDERS.get(key) == name:
            return name
    return ""


# --- preparing ------------------------------------------------------------------------------------

def _version_id(source: dict) -> int:
    """The Civitai version a prepare_download call names: version_id, a model or version URL, or a
    file hash (AutoV2 or SHA256)."""
    if source.get("version_id"):
        try:
            return int(source["version_id"])
        except (TypeError, ValueError):
            raise DownloadError("version_id must be a number.")
    url = str(source.get("url") or "").strip()
    if url:
        parsed = urlparse(url)
        if not re.fullmatch(r"(www\.)?civitai\.(com|green|red)", parsed.netloc.lower()):
            raise DownloadError("Only civitai.com links.")
        wanted = parse_qs(parsed.query).get("modelVersionId")
        if wanted and wanted[0].isdigit():
            return int(wanted[0])
        match = re.search(r"/models/(\d+)", parsed.path)
        if not match:
            raise DownloadError("That link isn't a Civitai model (civitai.com/models/<id>).")
        model = civitai._get_json(f"/models/{match.group(1)}")
        versions = (model or {}).get("modelVersions") or []
        if not versions:
            raise DownloadError("Civitai has no such model, or it has no versions.")
        return int(versions[0]["id"])  # the newest
    file_hash = str(source.get("hash") or "").strip()
    if file_hash:
        version = civitai._get_json(f"/model-versions/by-hash/{file_hash}")
        if not version:
            raise DownloadError(f"Civitai knows no file with hash {file_hash}.")
        return int(version["id"])
    raise DownloadError("Say which version: version_id, url (a Civitai model link) or hash.")


def _pick_file(files: list[dict]) -> dict:
    safe = [f for f in files if (f.get("metadata") or {}).get("format") == "SafeTensor"
            and f.get("type") in (None, "Model", "Pruned Model")]
    if not safe:
        raise DownloadError("That version has no SafeTensors file; other formats can run code, so they aren't "
                            "downloaded.")
    return next((f for f in safe if f.get("primary")), safe[0])


def prepare(conversation_id: str, source: dict, installed: dict | None = None) -> dict:
    """A download for the user to accept, stored on the conversation: {n, ...}. One the
    conversation already has for that version comes back as it is."""
    root = models_root()
    if root is None:
        raise DownloadError("Downloads aren't set up here (COMFY_MODELS_DIR).")
    version_id = _version_id(source)
    for existing in (conversations.get(conversation_id) or {}).get("downloads") or []:
        if existing["version_id"] == version_id and existing["status"] in ("proposed", "downloading", "done"):
            return existing
    version = civitai._get_json(f"/model-versions/{version_id}")
    if not version:
        raise DownloadError(f"Civitai has no version {version_id}.")
    model = version.get("model") or {}
    kind = TYPE_FOLDERS.get(str(model.get("type") or "").lower())
    if kind is None:
        raise DownloadError(f"Only checkpoints and LoRAs are downloaded, and this is a {model.get('type')}.")
    chosen = _pick_file(version.get("files") or [])
    folders = family_folders()
    family = _family(version.get("baseModel") or "", kind, folders)
    if kind == "checkpoints" and not family:
        # Families whose base models are diffusion models alone (their own text encoders and VAE).
        family = _family(version.get("baseModel") or "", "diffusion_models", folders)
        if family:
            kind = "diffusion_models"
    tags = []
    if kind == "loras" and version.get("modelId"):
        tags = (civitai._get_json(f"/models/{version['modelId']}") or {}).get("tags") or []
    name = file_name(model.get("name") or "", version.get("name") or "", kind, tags)
    if installed is not None:
        from .catalog import match_installed
        local = match_installed({"type": "checkpoint" if kind != "loras" else "lora", "name": model.get("name"),
                                 "file": chosen.get("name")}, installed)
        if local:
            raise DownloadError(f"It looks installed already, as {local}: use that.")
    record = {
        "version_id": version_id, "model_id": version.get("modelId"), "model_name": model.get("name") or "",
        "version_name": version.get("name") or "", "model_type": model.get("type"),
        "base_model": version.get("baseModel") or "", "civitai_file": chosen.get("name"),
        "size_bytes": int(float(chosen.get("sizeKB") or 0) * 1024),
        "sha256": str((chosen.get("hashes") or {}).get("SHA256") or "").lower(),
        "file_id": chosen.get("id"), "folder": kind, "family": family, "name": name,
        "status": "proposed", "created_at": datetime.now(timezone.utc).isoformat(),
    }
    stored = conversations.add_download(conversation_id, record)
    if stored is None:
        raise DownloadError("The conversation is gone.")
    logger.info("download prepared conversation_id=%s n=%s version=%s target=%s/%s/%s", conversation_id,
                stored["n"], version_id, kind, family, name)
    return stored


# --- accepting and downloading --------------------------------------------------------------------

def _target(record: dict, folder: str, family: str, name: str) -> Path:
    root = models_root()
    if root is None:
        raise DownloadError("Downloads aren't set up here (COMFY_MODELS_DIR).")
    allowed = ("loras",) if record["folder"] == "loras" else ("checkpoints", "diffusion_models")
    if folder not in allowed:
        raise DownloadError(f"A {record['model_type']} goes in {' or '.join(allowed)}.")
    if family not in family_folders().get(folder, []):
        raise DownloadError(f"There's no family folder {family!r} in {folder}.")
    name = re.sub(r"\.safetensors$", "", (name or "").strip(), flags=re.IGNORECASE)
    if not NAME.match(name) or len(name) > MAX_NAME_LENGTH:
        raise DownloadError("The name may have lowercase letters, digits, dots, dashes and underscores only.")
    target = (root / folder / family / f"{name}.safetensors").resolve()
    if not target.is_relative_to(root.resolve()):
        raise DownloadError("That name leads out of the models folder.")
    if target.exists():
        raise DownloadError(f"There's a file called {target.name} there already.")
    return target


def accept(conversation_id: str, n: int, folder: str | None = None, family: str | None = None,
           name: str | None = None) -> dict:
    """The user accepted download n (with the folder, family and name they may have changed): it starts."""
    record = conversations.get_download(conversation_id, n)
    if record is None:
        raise LookupError(n)
    if record["status"] not in ("proposed", "failed", "cancelled", "declined"):
        raise DownloadError("That download is under way or done.")
    folder, family, name = folder or record["folder"], family or record["family"], name or record["name"]
    target = _target(record, folder, family, name)
    part = _part(target)
    # What came down before goes where it now goes, if the user changed the folder or the name.
    earlier = _part_of(record)
    if earlier is not None and earlier != part and earlier.is_file() and not part.exists():
        earlier.rename(part)
    kept = part.stat().st_size if part.is_file() else 0
    free = shutil.disk_usage(target.parent).free
    if record["size_bytes"] and free < record["size_bytes"] - kept + SPACE_MARGIN:
        raise DownloadError(f"Not enough free space: it needs {(record['size_bytes'] - kept) / 1e9:.1f} GB more and "
                            f"leaves {free / 1e9:.1f} GB.")
    stored = conversations.update_download(conversation_id, n, lambda d: d.update(
        status="downloading", folder=folder, family=family, name=target.stem, error=None, kept_bytes=None,
        accepted_at=datetime.now(timezone.utc).isoformat()))
    with _lock:
        _progress[(conversation_id, n)] = {"bytes": kept, "total": record["size_bytes"]}
        _cancelled.discard((conversation_id, n))
    threading.Thread(target=_download, args=(conversation_id, n, record, target),
                     name=f"download-{conversation_id[:8]}-{n}", daemon=True).start()
    logger.info("download accepted conversation_id=%s n=%s target=%s", conversation_id, n, target)
    return stored


def decline(conversation_id: str, n: int) -> dict:
    record = conversations.get_download(conversation_id, n)
    if record is None:
        raise LookupError(n)
    if record["status"] != "proposed":
        raise DownloadError("Only a download not started yet can be turned down.")
    return conversations.update_download(conversation_id, n, lambda d: d.update(status="declined", told=False))


def cancel(conversation_id: str, n: int) -> None:
    with _lock:
        _cancelled.add((conversation_id, n))


def progress(conversation_id: str, n: int) -> dict | None:
    with _lock:
        found = _progress.get((conversation_id, n))
        return dict(found) if found else None


def _open(url: str, offset: int = 0):
    """The file's response, streaming, from byte `offset` on (tests replace it)."""
    headers = {"User-Agent": "prompt-enhancer"}
    if offset:
        headers["Range"] = f"bytes={offset}-"  # kept on the redirect: the file's host answers 206 with the rest
    key = get_settings().civitai_api_key
    if key:
        headers["Authorization"] = f"Bearer {key}"  # dropped by requests on the redirect to another host
    return requests.get(url, headers=headers, stream=True, timeout=TIMEOUT, allow_redirects=True)


def _part(target: Path) -> Path:
    return target.with_name(target.name + ".part")


def _part_of(record: dict) -> Path | None:
    """Where a download's file is being written, by its record's folder, family and name."""
    root = models_root()
    if root is None or not record.get("family") or not record.get("name"):
        return None
    return root / record["folder"] / record["family"] / f"{record['name']}.safetensors.part"


def _check_header(path: Path) -> None:
    """A SafeTensors file starts with its header's length (8 bytes, little-endian) and a JSON object."""
    with open(path, "rb") as f:
        head = f.read(9)
    if len(head) < 9 or head[8:9] != b"{" or struct.unpack("<Q", head[:8])[0] > 100_000_000:
        raise DownloadError("What came down isn't a SafeTensors file.")


def _fetch(key: tuple, url: str, part: Path, got: dict) -> None:
    """One try: what's missing of the file into `part`, from `got["bytes"]` on (the whole file when
    the host doesn't take a range). got: {"bytes", "total", "digest"}, kept up to date."""
    with _open(url, got["bytes"]) as response:
        if response.status_code in (401, 403):
            raise DownloadError("Civitai asks to sign in for it (CIVITAI_API_KEY), or it is early access.")
        if response.status_code == 429 or response.status_code >= 500:
            raise _Cut(f"Civitai answered {response.status_code}")  # busy or down for a moment: try again
        if response.status_code == 416 and got["bytes"]:
            return  # nothing left to send: it is whole already (checked after)
        if got["bytes"] and response.status_code == 200:  # the range wasn't taken: from the start again
            got.update(bytes=0, digest=hashlib.sha256())
        elif response.status_code not in (200, 206):
            raise DownloadError(f"Civitai answered {response.status_code}.")
        whole = re.search(r"/(\d+)$", response.headers.get("Content-Range") or "")
        length = int(response.headers.get("Content-Length") or 0)
        got["total"] = int(whole.group(1)) if whole else (got["bytes"] + length if length else got["total"])
        with _lock:
            _progress[key] = {"bytes": got["bytes"], "total": got["total"]}
        with open(part, "ab" if got["bytes"] else "wb") as out:
            for chunk in response.iter_content(CHUNK):
                if key in _cancelled:
                    raise _Cancelled()
                out.write(chunk)
                got["digest"].update(chunk)
                got["bytes"] += len(chunk)
                with _lock:
                    _progress[key]["bytes"] = got["bytes"]
    if got["total"] and got["bytes"] < got["total"]:
        raise _Cut(f"{got['bytes']} of {got['total']} bytes")


def _wait_to_retry(key: tuple, seconds: int, attempt: int) -> None:
    with _lock:
        _progress[key] = {**_progress.get(key, {}), "retry": {"attempt": attempt, "of": len(RETRY_WAITS),
                                                               "seconds": seconds}}
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if key in _cancelled:
            raise _Cancelled()
        time.sleep(0.2)
    with _lock:
        _progress.get(key, {}).pop("retry", None)


def _kept_so_far(part: Path) -> dict:
    """What came down in earlier tries (a part file), with its SHA256 so far."""
    digest = hashlib.sha256()
    size = 0
    if part.is_file():
        with open(part, "rb") as f:
            while chunk := f.read(CHUNK):
                digest.update(chunk)
                size += len(chunk)
    return {"bytes": size, "total": 0, "digest": digest}


def _download(conversation_id: str, n: int, record: dict, target: Path) -> None:
    key = (conversation_id, n)
    part = _part(target)
    url = f"{DOWNLOAD_URL}{record['version_id']}?type=Model&format=SafeTensor"
    try:
        with _one_at_a_time:
            # Civitai's size is an estimate (its sizeKB): the file's host says the exact one. A part that
            # isn't this file fails the checksum after, and goes then.
            if part.is_file():
                with _lock:
                    _progress[key] = {**_progress.get(key, {}), "checking": True}  # hashing what's there
            got = {**_kept_so_far(part), "total": record["size_bytes"] or 0}
            for attempt in range(len(RETRY_WAITS) + 1):
                try:
                    _fetch(key, url, part, got)
                    break
                except (requests.RequestException, _Cut) as e:  # the connection failed: carry on from there
                    logger.info("download cut conversation_id=%s n=%s at=%s error=%r", conversation_id, n,
                                got["bytes"], e)
                    if attempt == len(RETRY_WAITS):
                        raise _Kept(f"The connection kept failing ({e.__class__.__name__}). What came down is "
                                    f"kept: download it again to carry on from there.")
                    _wait_to_retry(key, RETRY_WAITS[attempt], attempt + 1)
            digest = got["digest"]
            _check_header(part)
            if record["sha256"] and digest.hexdigest() != record["sha256"]:
                raise DownloadError("The file doesn't match Civitai's checksum: it was thrown away.")
            if target.exists():
                raise DownloadError(f"A file called {target.name} appeared meanwhile: this one was thrown away.")
            part.rename(target)
        relative = str(target.relative_to(models_root() / record_folder(target)))
        conversations.update_download(conversation_id, n, lambda d: d.update(
            status="done", installed_as=relative, finished_at=datetime.now(timezone.utc).isoformat(), told=False))
        logger.info("download done conversation_id=%s n=%s file=%s", conversation_id, n, target)
        _refresh_catalog()
    except _Cancelled:
        part.unlink(missing_ok=True)
        conversations.update_download(conversation_id, n, lambda d: d.update(status="cancelled", told=False))
        logger.info("download cancelled conversation_id=%s n=%s", conversation_id, n)
    except _Kept as e:
        kept = part.stat().st_size if part.is_file() else 0
        conversations.update_download(conversation_id, n, lambda d: d.update(
            status="failed", error=str(e), kept_bytes=kept, told=False))
        logger.warning("download failed, kept conversation_id=%s n=%s bytes=%d", conversation_id, n, kept)
    except Exception as e:
        part.unlink(missing_ok=True)
        detail = str(e) if isinstance(e, DownloadError) else f"The download failed: {e.__class__.__name__}."
        conversations.update_download(conversation_id, n, lambda d: d.update(
            status="failed", error=detail, kept_bytes=None, told=False))
        logger.warning("download failed conversation_id=%s n=%s error=%r", conversation_id, n, e)
    finally:
        with _lock:
            _progress.pop(key, None)
            _cancelled.discard(key)


def record_folder(target: Path) -> str:
    """The models folder a target sits in (loras, checkpoints, diffusion_models)."""
    return target.parent.parent.name


def _refresh_catalog() -> None:
    client = comfy.client()
    if client is None:
        return
    try:
        from . import catalog
        catalog.refresh_files(client)
    except Exception as e:  # the next turn lists the files anyway
        logger.info("catalog refresh after a download failed: %r", e)


def after_restart() -> None:
    """Downloads cut short by a restart: failed, keeping what came down, to carry on from there when
    accepted again."""
    for summary in conversations.list_summaries():
        conversation = conversations.get(summary["id"]) or {}
        for record in conversation.get("downloads") or []:
            if record["status"] != "downloading":
                continue
            part = _part_of(record)
            kept = part.stat().st_size if part is not None and part.is_file() else 0
            conversations.update_download(summary["id"], record["n"], lambda d: d.update(
                status="failed", told=False, kept_bytes=kept or None,
                error="The app restarted during the download." + (
                    " What came down is kept: download it again to carry on from there." if kept else "")))


def news_text(records: list[dict]) -> str | None:
    """For the turn context: what happened to the downloads the bot prepared since it last heard."""
    lines = []
    for d in records:
        label = f"Download {d['n']} ({d['model_name']} {d['version_name']}, {d['model_type']})"
        if d["status"] == "done":
            lines.append(f"- {label}: the user downloaded it; installed as {d['folder']}\\{d['installed_as']}.")
        elif d["status"] == "declined":
            lines.append(f"- {label}: the user turned it down.")
        elif d["status"] == "cancelled":
            lines.append(f"- {label}: the user cancelled it while it downloaded.")
        elif d["status"] == "failed":
            lines.append(f"- {label}: it failed: {d.get('error')}")
    return "## Downloads Since Your Last Reply\n" + "\n".join(lines) if lines else None

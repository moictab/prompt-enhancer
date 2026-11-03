"""Images shared from Civitai by URL: the image itself and what Civitai says about how it
was made, so the Conversar bot can reproduce it or borrow its style.

The user pastes a link like https://civitai.com/images/147084007 (or on civitai.green or
civitai.red) in a message; the app (never
the bot: it has no way to fetch anything) reads the image's record from Civitai's public API
(`/images?imageId=…&withMeta=true`), downloads the image, and resolves the resources it was
made with (`/model-versions/{id}` or `/model-versions/by-hash/{hash}`): checkpoint and LoRAs
with their weights, base model, trigger words and file names. The image then joins the
conversation like one the user shared, with that as its metadata -- and, when it was made
with ComfyUI, its workflow, runnable as image:N.

Civitai keeps generation data in three shapes, all read: A1111/Forge (`resources` with
hashes), its own generator (`civitaiResources` by version id), ComfyUI (`comfy`, the graph,
plus `additionalResources`). CIVITAI_API_KEY is sent to the API when set.
"""
import json
import logging
import re
from dataclasses import dataclass

import requests

from .comfy_graph import is_api_workflow
from .config import get_settings

logger = logging.getLogger("app.civitai")

API = "https://civitai.com/api/v1"
# civitai.com, and its sister sites civitai.green (safe for work) and civitai.red (adult content):
# one site behind them, so an image id is the same on all three.
IMAGE_URL = re.compile(r"https?://(?:www\.)?civitai\.(?:com|green|red)/images/(\d+)", re.IGNORECASE)
TIMEOUT = (10, 60)
MAX_IMAGE_BYTES = 20 * 1024 * 1024
MAX_LOOKUPS = 10
MAX_TRIGGER_WORDS = 6
CONTENT_TYPES = ("image/png", "image/jpeg", "image/webp")

_versions: dict[str, dict | None] = {}


class CivitaiError(RuntimeError):
    pass


@dataclass
class SharedImage:
    data: bytes
    content_type: str
    metadata: dict
    workflow: dict | None


def image_ids(text: str) -> list[int]:
    """The Civitai image ids linked in a message, in order, without repeats."""
    return list(dict.fromkeys(int(m.group(1)) for m in IMAGE_URL.finditer(text or "")))


def _headers() -> dict:
    headers = {"User-Agent": "prompt-enhancer"}
    key = get_settings().civitai_api_key
    if key:
        headers["Authorization"] = f"Bearer {key}"
    return headers


def _get_json(path: str, params: dict | None = None):
    try:
        response = requests.get(f"{API}{path}", params=params, headers=_headers(), timeout=TIMEOUT)
    except requests.exceptions.RequestException as e:
        raise CivitaiError(f"Civitai didn't answer: {e.__class__.__name__}")
    if response.status_code == 404:
        return None
    if response.status_code != 200:
        raise CivitaiError(f"Civitai answered {response.status_code}")
    try:
        return response.json()
    except ValueError:
        raise CivitaiError("Civitai sent something that isn't JSON")


def _version(version_id=None, hash_=None) -> dict | None:
    key = f"id:{version_id}" if version_id else f"hash:{str(hash_).lower()}"
    if key not in _versions:
        try:
            path = f"/model-versions/{int(version_id)}" if version_id else f"/model-versions/by-hash/{hash_}"
            _versions[key] = _get_json(path)
        except (CivitaiError, ValueError) as e:
            logger.info("civitai resource lookup failed key=%s error=%s", key, e)
            return None
    return _versions[key]


def _number(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return int(number) if number.is_integer() else round(number, 3)


def _resources(meta: dict) -> list[dict]:
    """The checkpoint, LoRAs and so on the image was made with, resolved against Civitai."""
    found = []
    for r in meta.get("civitaiResources") or []:
        if isinstance(r, dict):
            found.append({"type": r.get("type"), "weight": r.get("weight", r.get("strength")),
                          "version_id": r.get("modelVersionId")})
    for r in meta.get("resources") or []:
        if isinstance(r, dict):
            found.append({"type": r.get("type"), "weight": r.get("weight"), "name": r.get("name"), "hash": r.get("hash")})
    for r in meta.get("additionalResources") or []:
        if isinstance(r, dict):
            found.append({"type": r.get("type"), "weight": r.get("strength", r.get("weight")), "name": r.get("name")})
    resources, seen, lookups = [], set(), 0
    for r in found:
        version = None
        if (r.get("version_id") or r.get("hash")) and lookups < MAX_LOOKUPS:
            lookups += 1
            version = _version(r.get("version_id"), r.get("hash"))
        entry = {"type": str(r.get("type") or "").lower() or None}
        if version:
            model = version.get("model") or {}
            files = version.get("files") or []
            primary = next((f for f in files if f.get("primary")), files[0] if files else {})
            entry.update(type=str(model.get("type") or entry["type"] or "").lower() or None,
                         name=model.get("name"), version=version.get("name"), base_model=version.get("baseModel"),
                         trigger_words=[w for w in version.get("trainedWords") or [] if isinstance(w, str)][
                             :MAX_TRIGGER_WORDS],
                         file=primary.get("name"))
        elif r.get("name"):
            entry["name"] = r["name"]
        else:
            continue
        weight = _number(r.get("weight"))
        if weight is not None and entry["type"] not in ("checkpoint", "model", "upscaler", "vae"):
            entry["weight"] = weight
        key = (entry.get("name"), entry.get("version"), entry.get("file"))
        if key in seen:
            continue
        seen.add(key)
        resources.append({k: v for k, v in entry.items() if v not in (None, "", [])})
    return resources


def _workflow(meta: dict) -> dict | None:
    comfy = meta.get("comfy")
    if isinstance(comfy, str):
        try:
            comfy = json.loads(comfy)
        except ValueError:
            return None
    if isinstance(comfy, dict):
        graph = comfy.get("prompt", comfy)
        return graph if is_api_workflow(graph) else None
    return None


def metadata_from(item: dict, meta: dict, resources: list[dict], has_workflow: bool) -> dict:
    """Civitai's record of an image, in the shape image_metadata.read gives a file's metadata."""
    settings = {}
    for key, name in (("seed", "seed"), ("steps", "steps"), ("cfgScale", "cfg"), ("sampler", "sampler"),
                      ("scheduler", "scheduler"), ("Schedule type", "scheduler"), ("denoise", "denoise"),
                      ("clipSkip", "clip skip")):
        if meta.get(key) not in (None, "") and name not in settings:
            settings[name] = meta[key]
    if item.get("baseModel"):
        settings["base model"] = item["baseModel"]
    metadata = {
        "format": None,
        "width": item.get("width"),
        "height": item.get("height"),
        "generator": "Civitai",
        "positive": meta.get("prompt"),
        "negative": meta.get("negativePrompt"),
        "settings": settings,
        "resources": resources,
        "source": {"site": "Civitai", "image_id": item.get("id"), "url": f"https://civitai.com/images/{item.get('id')}",
                   "author": item.get("username")},
    }
    if has_workflow:
        metadata["has_workflow"] = True
    if not resources and meta.get("Model"):
        metadata["models"] = [meta["Model"]]
    return {k: v for k, v in metadata.items() if v not in (None, "", [], {})}


def fetch_image(image_id: int) -> SharedImage:
    """The image and its metadata, or CivitaiError with what went wrong."""
    data = _get_json("/images", {"imageId": image_id, "withMeta": "true", "nsfw": "X"})
    items = (data or {}).get("items") or []
    item = next((i for i in items if i.get("id") == image_id), None)
    if item is None:
        raise CivitaiError(f"Civitai has no image {image_id} (or it isn't public)")
    if item.get("type") != "image":
        raise CivitaiError(f"Civitai item {image_id} is a {item.get('type')}, not an image")
    try:
        response = requests.get(item["url"], headers={"User-Agent": "prompt-enhancer"}, timeout=TIMEOUT)
    except requests.exceptions.RequestException as e:
        raise CivitaiError(f"Couldn't download Civitai image {image_id}: {e.__class__.__name__}")
    content_type = response.headers.get("Content-Type", "").split(";")[0].strip()
    if response.status_code != 200 or content_type not in CONTENT_TYPES:
        raise CivitaiError(f"Couldn't download Civitai image {image_id} ({response.status_code}, {content_type})")
    if len(response.content) > MAX_IMAGE_BYTES:
        raise CivitaiError(f"Civitai image {image_id} is larger than {MAX_IMAGE_BYTES // (1024 * 1024)} MB")
    meta = item.get("meta") if isinstance(item.get("meta"), dict) else {}
    if isinstance(meta.get("meta"), dict):
        meta = meta["meta"]  # asked by imageId, the API nests it one level deeper than in its listings
    workflow = _workflow(meta)
    metadata = metadata_from(item, meta, _resources(meta), workflow is not None)
    metadata["format"] = content_type.split("/")[1].upper()
    logger.info("civitai image fetched image_id=%s bytes=%d resources=%d workflow=%s",
                image_id, len(response.content), len(metadata.get("resources") or []), workflow is not None)
    return SharedImage(response.content, content_type, metadata, workflow)

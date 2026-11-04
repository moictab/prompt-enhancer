"""What an image file says about how it was made, and a smaller copy of it for the LLM.

Generators embed their settings in the file:
- ComfyUI writes PNG text chunks "prompt" (the API-format graph it ran) and "workflow" (the
  editor's graph); its WebP/JPEG savers put them in EXIF as "prompt:{...}" / "workflow:{...}".
- A1111, Forge, Civitai and many others write a "parameters" text: the prompt, a
  "Negative prompt: ..." line, then "Steps: 30, Sampler: ..., Seed: ...". JPEG and WebP
  carry it in EXIF UserComment.

read() never raises: a damaged or foreign file just yields less. Only files picked or
dropped keep this; an image pasted from the clipboard has been re-encoded without it.
"""
import io
import json
import logging
import re

from PIL import Image, PngImagePlugin

from .comfy_graph import Graph, GraphError, is_api_workflow, is_editor_workflow

logger = logging.getLogger("app.image_metadata")

# ComfyUI graphs can be large; Pillow's default cap for a PNG text chunk is 1 MB.
PngImagePlugin.MAX_TEXT_CHUNK = 16 * 1024 * 1024

# The copy sent to the LLM and shown as a thumbnail: models scale larger images down anyway.
SMALL_MAX_SIDE = 1568
SMALL_MAX_BYTES = 1_500_000
MAX_PROMPT_CHARS = 4000

_EXIF_IFD = 0x8769
_USER_COMMENT = 0x9286
_TEXT_TAGS = (0x010E, 0x010F, 0x0110, 0x013B, 0x9C9C)  # ImageDescription, Make, Model, Artist, XPComment
_LORA_TAG = re.compile(r"<lora:([^:>]+):([-0-9.]+)[^>]*>", re.IGNORECASE)
_PARAM_KEYS = {"Model": "models", "VAE": "models"}


def _decode_user_comment(raw) -> str:
    if isinstance(raw, str):
        return raw
    if not isinstance(raw, bytes):
        return ""
    head, body = raw[:8], raw[8:]
    if head.startswith(b"UNICODE"):
        candidates = []
        for encoding in ("utf-16-be", "utf-16-le"):
            try:
                text = body.decode(encoding)
            except UnicodeDecodeError:
                continue
            printable = sum(1 for c in text if c.isprintable() or c in "\n\r\t")
            candidates.append((printable, text))
        return max(candidates)[1].rstrip("\x00") if candidates else ""
    if head.startswith(b"ASCII") or head == b"\x00" * 8:
        return body.decode("utf-8", "replace").rstrip("\x00")
    return raw.decode("utf-8", "replace").rstrip("\x00")


def _exif_texts(image: Image.Image) -> list[str]:
    try:
        exif = image.getexif()
    except Exception:
        return []
    texts = []
    for tag in _TEXT_TAGS:
        value = exif.get(tag)
        if isinstance(value, bytes):
            value = value.decode("utf-16-le" if tag == 0x9C9C else "utf-8", "replace")
        if isinstance(value, str) and value.strip():
            texts.append(value.strip("\x00").strip())
    try:
        comment = exif.get_ifd(_EXIF_IFD).get(_USER_COMMENT)
    except Exception:
        comment = None
    comment = _decode_user_comment(comment).strip()
    if comment:
        texts.append(comment)
    return texts


def _json(text: str):
    try:
        return json.loads(text)
    except (ValueError, TypeError):
        return None


def _scan_value(text: str, i: int) -> tuple[str, int]:
    """One value of a "Key: value, Key: value" line: quoted, bracketed (JSON), or up to the next comma."""
    if i < len(text) and text[i] == '"':
        j, escaped = i + 1, False
        while j < len(text):
            if escaped:
                escaped = False
            elif text[j] == "\\":
                escaped = True
            elif text[j] == '"':
                break
            j += 1
        return text[i + 1:j], j + 1
    if i < len(text) and text[i] in "[{":
        depth, j, quoted = 0, i, False
        while j < len(text):
            c = text[j]
            if c == '"' and (j == 0 or text[j - 1] != "\\"):
                quoted = not quoted
            elif not quoted and c in "[{":
                depth += 1
            elif not quoted and c in "]}":
                depth -= 1
                if depth == 0:
                    return text[i:j + 1], j + 1
            j += 1
        return text[i:], len(text)
    j = text.find(",", i)
    j = len(text) if j == -1 else j
    return text[i:j].strip(), j


def parse_settings_line(line: str) -> dict:
    settings, i = {}, 0
    while i < len(line):
        match = re.compile(r"\s*([A-Za-z][\w \-/().]*?):\s*").match(line, i)
        if not match:
            break
        value, i = _scan_value(line, match.end())
        settings[match.group(1).strip()] = value
        while i < len(line) and line[i] in ", ":
            i += 1
    return settings


def parse_parameters(text: str) -> dict:
    """A1111-style "parameters": prompt lines, an optional "Negative prompt:", then the settings line."""
    lines = text.strip().split("\n")
    settings_index = next((i for i in range(len(lines) - 1, -1, -1) if lines[i].startswith("Steps:")), None)
    if settings_index is None and len(lines) > 1 and len(parse_settings_line(lines[-1])) >= 3:
        settings_index = len(lines) - 1
    settings = parse_settings_line(" ".join(lines[settings_index:])) if settings_index is not None else {}
    body = lines[:settings_index] if settings_index is not None else lines
    negative_index = next((i for i, line in enumerate(body) if line.startswith("Negative prompt:")), None)
    positive = "\n".join(body[:negative_index] if negative_index is not None else body).strip()
    negative = ""
    if negative_index is not None:
        negative = "\n".join(body[negative_index:])[len("Negative prompt:"):].strip()
    result = {"positive": positive, "negative": negative, "settings": settings}
    result["models"] = [settings[k] for k in _PARAM_KEYS if settings.get(k)]
    result["loras"] = [{"name": m.group(1), "strength": _number(m.group(2))}
                       for m in _LORA_TAG.finditer(positive)]
    return result


def _number(text: str):
    try:
        value = float(text)
    except ValueError:
        return text
    return int(value) if value.is_integer() else value


def _comfy_graph(candidates: list) -> dict | None:
    for value in candidates:
        if isinstance(value, dict) and is_api_workflow(value):
            return value
        if isinstance(value, dict) and is_api_workflow(value.get("prompt")):
            return value["prompt"]
    return None


def read(data) -> tuple[dict | None, dict | None]:
    """(metadata, workflow): the generation metadata worth showing, or None if the file has
    none, and the API-format ComfyUI workflow it embeds, if any (stored apart: it can be large).
    data: the image's bytes, or a file's path (only its header is read: the text chunks Pillow
    writes come before the pixels)."""
    try:
        with Image.open(io.BytesIO(data) if isinstance(data, (bytes, bytearray)) else data) as image:
            texts = {k: v for k, v in image.info.items() if isinstance(v, str)}
            exif_texts = _exif_texts(image)
            size = image.size
            image_format = image.format
    except Exception as e:
        logger.info("image metadata unreadable error=%r", e)
        return None, None

    prompt_text = texts.get("prompt")
    workflow_text = texts.get("workflow")
    parameters = texts.get("parameters")
    for value in exif_texts:
        if value.startswith("prompt:"):
            prompt_text = prompt_text or value[len("prompt:"):]
        elif value.startswith("workflow:"):
            workflow_text = workflow_text or value[len("workflow:"):]
        elif value.startswith("{"):
            prompt_text = prompt_text or value
        elif not parameters and ("Steps:" in value or "Negative prompt:" in value):
            parameters = value

    workflow = _comfy_graph([_json(prompt_text)] if prompt_text else [])
    metadata = {"format": image_format, "width": size[0], "height": size[1]}
    found = False
    if workflow is not None:
        try:
            info = Graph(workflow).generation_info()
        except GraphError:
            info = {}
        metadata.update({"generator": "ComfyUI", "has_workflow": True, **info})
        found = True
    elif workflow_text and is_editor_workflow(_json(workflow_text)):
        metadata.update({"generator": "ComfyUI", "editor_workflow_only": True})
        found = True
    if parameters:
        parsed = parse_parameters(parameters)
        if workflow is None:
            metadata.update({"generator": metadata.get("generator") or "A1111", **parsed})
        else:
            metadata["parameters"] = parsed["settings"]
        found = True
    if not found:
        return None, None
    for key in ("positive", "negative"):
        if isinstance(metadata.get(key), str) and len(metadata[key]) > MAX_PROMPT_CHARS:
            metadata[key] = metadata[key][:MAX_PROMPT_CHARS] + "..."
    return {k: v for k, v in metadata.items() if v not in ("", [], {}, None)}, workflow


def _format_value(value) -> str:
    return f"{value:g}" if isinstance(value, float) else str(value)


def _short(text: str, limit: int = 300) -> str:
    return text if len(text) <= limit else text[:limit] + "..."


def _resource_text(resource: dict, installed: dict | None = None) -> str:
    name = resource.get("name") or resource.get("file") or "?"
    text = f"{resource.get('type') or 'resource'} \"{name}\""
    if resource.get("version"):
        text += f" {resource['version']}"
    details = []
    if resource.get("base_model"):
        details.append(f"base {resource['base_model']}")
    if resource.get("weight") is not None:
        details.append(f"weight {_format_value(resource['weight'])}")
    if resource.get("trigger_words"):
        details.append(f"trigger words: {', '.join(resource['trigger_words'])}")
    if resource.get("file") and resource.get("file") != name:
        details.append(f"file {resource['file']}")
    if installed is not None:
        from .catalog import match_installed
        local = match_installed(resource, installed)
        details.append(f"installed here as {local}" if local else "no installed file matches its name")
    if resource.get("version_id"):
        details.append(f"Civitai version {resource['version_id']}")
    return f"{text} ({'; '.join(details)})" if details else text


def describe(metadata: dict, number: int, installed: dict | None = None) -> str:
    """installed: the catalog's entries, to say which of a Civitai image's resources the user has."""
    """The metadata as text for the LLM (English, like everything it reads)."""
    head = f"Image {number} metadata ({metadata.get('format', 'image')}, {metadata.get('width')}x{metadata.get('height')}"
    if metadata.get("generator"):
        head += f", made with {metadata['generator']}"
    if metadata.get("has_workflow"):
        head += f'; its ComfyUI workflow can be used as "image:{number}"'
    elif metadata.get("editor_workflow_only"):
        head += "; it embeds only the ComfyUI editor graph, which can't be run from here"
    lines = [head + "):"]
    source = metadata.get("source")
    if source:
        by = f", by {source['author']}" if source.get("author") else ""
        lines.append(f"- Source: {source.get('site')} image {source.get('image_id')}{by}")
    if metadata.get("positive"):
        lines.append(f"- Positive: {metadata['positive']}")
    if metadata.get("negative"):
        lines.append(f"- Negative: {metadata['negative']}")
    if metadata.get("models"):
        lines.append(f"- Models: {', '.join(metadata['models'])}")
    if metadata.get("loras"):
        loras = []
        for lora in metadata["loras"]:
            strengths = [_format_value(s) for s in (lora.get("strength"), lora.get("strength_clip")) if s is not None]
            if len(strengths) == 2 and strengths[0] == strengths[1]:
                strengths = strengths[:1]
            loras.append(f"{lora['name']} ({'/'.join(strengths)})" if strengths else lora["name"])
        lines.append(f"- LoRAs: {', '.join(loras)}")
    if metadata.get("resources"):
        lines.append(f"- {metadata.get('generator', 'Source')} resources: "
                     + "; ".join(_resource_text(r, installed) for r in metadata["resources"]))
    settings = {**(metadata.get("parameters") or {}), **(metadata.get("settings") or {})}
    if settings:
        values = [f"{k}: {_short(_format_value(v))}" for k, v in settings.items()]
        lines.append("- Settings: " + ", ".join(values))
    return "\n".join(lines)


def small_copy(data: bytes) -> bytes | None:
    """A JPEG no larger than SMALL_MAX_SIDE, or None when the original is small enough already."""
    try:
        image = Image.open(io.BytesIO(data))
        if max(image.size) <= SMALL_MAX_SIDE and len(data) <= SMALL_MAX_BYTES:
            return None
        image.thumbnail((SMALL_MAX_SIDE, SMALL_MAX_SIDE))
        if image.mode not in ("RGB", "L"):
            background = Image.new("RGB", image.size, (255, 255, 255))
            rgba = image.convert("RGBA")
            background.paste(rgba, mask=rgba.split()[-1])
            image = background
        out = io.BytesIO()
        image.save(out, "JPEG", quality=88)
        return out.getvalue()
    except Exception as e:
        logger.info("image small copy failed error=%r", e)
        return None

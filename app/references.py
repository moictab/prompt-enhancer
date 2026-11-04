"""Reference folders: folders of images on this PC that the user accepted in Ajustes, so a
message can point the bot at one, for style and for writing prompts. The message names it as
"@name", or pastes the folder's path or the path of a folder inside it.

The app reads them, never the bot:
- It walks the folder, taking images only (PNG, JPEG, WebP) and skipping links that lead out
  of it, and writes nothing there.
- It reads the generation metadata of up to MAX_STUDIED images (image_metadata.read: ComfyUI's
  workflow or A1111's parameters).
- It attaches a study to that message (study): what recurs across the folder (base models,
  LoRAs and their weights, sampler settings, proportions, the phrases of the prompts, a few
  whole prompts chosen to differ), plus contact sheets (numbered thumbnails of a spread of the
  folder) to see the look.

The user picked these images, so what recurs is what works for them. The bot draws its
conclusions and keeps them as notes. Once it has answered, the study is replayed as one line
(answered_text), as the balance is (balance.py).
"""
import io
import logging
import os
import re
import statistics
import threading
import uuid
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps

from . import image_metadata, storage
from .comfy_graph import Graph, GraphError, _nearest_aspect, _orientation
from .config import get_settings

logger = logging.getLogger(__name__)

IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp")
MAX_LISTED = 5000  # files looked at in a folder
MAX_STUDIED = 300  # images whose metadata a study reads: an even spread when there are more
MAX_NAME_LENGTH = 60
SHEET_COLUMNS, SHEET_ROWS, TILE = 4, 4, 320  # a contact sheet: 1280x1280, 16 thumbnails
MAX_SHEETS = 3
SAMPLE_PROMPTS = 10
TOP_PHRASES = 60
TOP_NEGATIVE = 20
TOP_VALUES = 6
MAX_SAMPLE_CHARS = 1500
MAX_PHRASE_WORDS = 8

_lock = threading.Lock()


class ReferenceFolderError(ValueError):
    """Something the user can fix: shown to them as is."""


def _default_path() -> str:
    return os.path.join(get_settings().data_dir, "reference_folders.json")


def list_folders(path: str | None = None) -> list[dict]:
    return storage.read_json(path or _default_path(), default=[])


def _folder_path(text: str) -> Path:
    raw = (text or "").strip().strip('"').strip("'")
    if not raw:
        raise ReferenceFolderError("Write the folder's path.")
    folder = Path(raw).expanduser()
    if not folder.is_absolute():
        raise ReferenceFolderError(f"Write the whole path, from the drive (C:\\...): {raw}")
    if not folder.is_dir():
        raise ReferenceFolderError(f"There's no folder at {raw}.")
    return folder.resolve()


def create_folder(name: str, folder_path: str, path: str | None = None) -> dict:
    name = " ".join((name or "").split())
    if not name or len(name) > MAX_NAME_LENGTH:
        raise ReferenceFolderError(f"Give it a name of up to {MAX_NAME_LENGTH} characters.")
    if "@" in name:
        raise ReferenceFolderError("The name can't have an @: messages write it as @name.")
    folder = _folder_path(folder_path)
    with _lock:
        folders = list_folders(path)
        if any(f["name"].lower() == name.lower() for f in folders):
            raise ReferenceFolderError(f"There's already a folder called {name}.")
        same = next((f for f in folders if Path(f["path"]) == folder), None)
        if same:
            raise ReferenceFolderError(f"That folder is already accepted, as {same['name']}.")
        entry = {"id": uuid.uuid4().hex, "name": name, "path": str(folder),
                 "created_at": datetime.now(timezone.utc).isoformat()}
        storage.write_json(path or _default_path(), [*folders, entry])
    return entry


def delete_folder(folder_id: str, path: str | None = None) -> bool:
    with _lock:
        folders = list_folders(path)
        kept = [f for f in folders if f["id"] != folder_id]
        if len(kept) == len(folders):
            return False
        storage.write_json(path or _default_path(), kept)
    return True


def images_in(directory: Path, root: Path) -> tuple[list[Path], bool]:
    """The images under directory, in path order, and whether there were more than MAX_LISTED.
    A link that leads out of the accepted folder (root) is skipped."""
    found: list[Path] = []
    real_root = root.resolve()
    for current, dirs, files in os.walk(directory):  # links to folders aren't followed
        dirs.sort()
        for name in sorted(files):
            if not name.lower().endswith(IMAGE_SUFFIXES):
                continue
            file = Path(current, name)
            try:
                if not file.resolve().is_relative_to(real_root):
                    continue
            except OSError:
                continue
            if len(found) == MAX_LISTED:
                return found, True
            found.append(file)
    return found, False


def with_counts(entry: dict) -> dict:
    """The folder as Ajustes lists it: how many images it has, or that it's gone."""
    folder = Path(entry["path"])
    if not folder.is_dir():
        return {**entry, "missing": True, "images": 0, "capped": False}
    files, capped = images_in(folder, folder)
    return {**entry, "missing": False, "images": len(files), "capped": capped}


# --- a message pointing at a folder -----------------------------------------------------------

def find_mention(text: str, folders: list[dict]) -> tuple[dict, Path] | None:
    """The accepted folder a message points at, and the folder within it to study: "@name"
    (the longest name that fits), or a pasted path to it or to a folder inside it. None when
    the message names none of them."""
    lowered = (text or "").lower()
    for entry in sorted(folders, key=lambda f: -len(f["name"])):
        if re.search(r"(?<![\w@])@" + re.escape(entry["name"].lower()) + r"(?!\w)", lowered):
            return entry, Path(entry["path"])
    normalized = (text or "").replace("/", "\\")
    low = normalized.lower()
    for entry in sorted(folders, key=lambda f: -len(f["path"])):
        root = entry["path"].replace("/", "\\").rstrip("\\")
        start = low.find(root.lower())
        while start != -1:
            end = start + len(root)
            if end == len(low) or not (low[end].isalnum() or low[end] in "_-"):
                rest = re.split(r"[\n\"'«»“”`]", normalized[end:])[0]
                return entry, _deepest(Path(entry["path"]), rest)
            start = low.find(root.lower(), start + 1)
    return None


def _deepest(root: Path, rest: str) -> Path:
    """The deepest existing folder inside root that `rest` (what follows root's path in the
    message) names: "\\poses\\noche, mira" -> root\\poses\\noche."""
    if not rest.startswith("\\"):
        return root
    rest = rest.lstrip("\\")
    real_root = root.resolve()
    cuts = sorted({m.start() for m in re.finditer(r"[\s,;:!?)\]]", rest)} | {len(rest)}, reverse=True)
    for cut in cuts:
        # A full stop ending the sentence isn't part of the path ("..", going up, is).
        candidate_text = re.sub(r"(?<!\.)\.$", "", rest[:cut].strip()).rstrip("\\")
        if not candidate_text:
            continue
        candidate = (root / candidate_text).resolve()
        if candidate.is_dir() and candidate.is_relative_to(real_root):
            return candidate
    return root


# --- the study ---------------------------------------------------------------------------------

INSTRUCTIONS = """What to do with it: these images were picked by the user as good examples of what they want, so \
what recurs across them is what works. Explore them and say what you conclude:
- the subject: the traits, features and wardrobe that define it, in the words the prompts use for them;
- the style: the look, medium, lighting and framing that recur (the contact sheets show them), and the words, \
LoRAs and models behind them;
- the recipe: base models, LoRAs and their weights, sampler settings and proportions that recur, and which of them \
are installed here;
- how the prompts are written: order, length, tags or sentences, quality words, weights, the negative.
Say how many images back each point, and what you would take from it for this conversation's model and family \
(their rules decide the prompt's form). Keep what is worth keeping as notes, named after the folder ("{name}: \
..."), telegraphic: what you would need to write such a prompt again without the folder. A hunch only as one. \
Unless the user asked for images too, don't launch anything in this reply."""

ANSWERED = ("[Earlier the user pointed you at the reference folder \"{name}\" ({images} images); the study went with "
            "that message and you answered it below. It isn't repeated: your reply and your notes hold what you took "
            "from it.]")


@dataclass
class Study:
    text: str
    sheets: list[bytes]  # contact sheets, JPEG
    counts: dict  # name, path, images, studied, with_metadata, sheets


def answered_text(counts: dict | None) -> str:
    counts = counts or {}
    return ANSWERED.format(name=counts.get("name", "?"), images=counts.get("images", "?"))


def _number(value):
    if isinstance(value, str):
        try:
            value = float(value)
        except ValueError:
            return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return value


def _fmt(value) -> str:
    return f"{value:g}" if isinstance(value, float) else str(value)


def _file_name(model: str) -> str:
    """A model or LoRA without its folder or extension: "illustrious\\wai_v14.safetensors" -> "wai_v14"."""
    name = str(model).replace("/", "\\").rsplit("\\", 1)[-1]
    return re.sub(r"\.(safetensors|ckpt|pt|pth|bin|gguf|sft)$", "", name, flags=re.IGNORECASE)


def _shape(width, height) -> str | None:
    if not (isinstance(width, int) and isinstance(height, int) and width > 0 and height > 0):
        return None
    aspect = _nearest_aspect(width, height)
    return f"{aspect or f'{width}x{height}'} {_orientation(width, height)}, {width * height / 1_000_000:.1f} MP"


def _read(file: Path, root: Path) -> dict:
    """One image: its path in the folder, its proportions, and what its metadata says."""
    item = {"file": file, "name": file.relative_to(root).as_posix() if file.is_relative_to(root) else file.name}
    try:
        with Image.open(file) as image:
            item["size"] = image.size
    except Exception as e:  # unreadable: left out of the study
        logger.info("reference image unreadable file=%s error=%r", file, e)
        return {}
    metadata, workflow = image_metadata.read(file)
    item["shape"] = _shape(*item["size"])
    if not metadata:
        return item
    settings = {**(metadata.get("parameters") or {}), **(metadata.get("settings") or {})}
    base_models = []
    if workflow:
        try:
            base_models = Graph(workflow).base_models()
        except GraphError:
            pass
    elif settings.get("Model"):
        base_models = [str(settings["Model"])]
    item.update({
        "generator": metadata.get("generator") or "unknown",
        "positive": metadata.get("positive") or "",
        "negative": metadata.get("negative") or "",
        "models": base_models,
        "loras": [(str(lora["name"]), _number(lora.get("strength"))) for lora in metadata.get("loras") or []
                  if lora.get("name")],
        "steps": settings.get("steps", settings.get("Steps")),
        "cfg": settings.get("cfg", settings.get("CFG scale")),
        "sampler": settings.get("sampler_name", settings.get("Sampler")),
        "scheduler": settings.get("scheduler", settings.get("Schedule type")),
    })
    return item


def _spread(items: list, count: int) -> list:
    """count items spread evenly over the list, in order."""
    if count <= 0:
        return []
    if len(items) <= count:
        return list(items)
    step = len(items) / count
    return [items[int(i * step)] for i in range(count)]


def phrases(prompt: str) -> set[str]:
    """The phrases of a prompt, as written apart (by commas, full stops, lines), lowercased and
    without weights or LoRA tags: "(red hair:1.2), freckles" -> {"red hair", "freckles"}."""
    text = re.sub(r"<lora:[^>]+>", " ", prompt or "", flags=re.IGNORECASE)
    text = re.sub(r"\(([^():]+):\s*[\d.]+\)", r"\1", text)
    text = re.sub(r"\bBREAK\b", ",", text)
    text = re.sub(r"[()\[\]{}]", " ", text)
    found = set()
    for part in re.split(r"[,\n;|]+|\.(?:\s|$)", text):
        phrase = " ".join(part.lower().split()).strip(" -:.")
        if phrase and len(phrase.split()) <= MAX_PHRASE_WORDS and len(phrase) <= 80:
            found.add(phrase)
    return found


def _tally(counter: Counter, top: int = TOP_VALUES, label=_fmt) -> str:
    shown = ", ".join(f"{label(value)} ({n})" for value, n in counter.most_common(top))
    rest = len(counter) - top
    return f"{shown}, and {rest} more" if rest > 0 else shown


def _installed(name: str, kind: str, installed: dict | None) -> str:
    if installed is None:
        return ""
    from .catalog import match_installed
    local = match_installed({"type": kind, "name": _file_name(name), "file": name}, installed)
    return f"installed here as {local}" if local else "no installed file matches its name"


def _recipe(item: dict) -> str:
    if "generator" not in item:
        return "no metadata"
    parts = [", ".join(_file_name(m) for m in item["models"]) or "model unknown"]
    parts += [f"{_file_name(name)} {_fmt(weight)}" if weight is not None else _file_name(name)
              for name, weight in item["loras"]]
    return "; ".join(parts)


def _pick_samples(items: list[dict], frequency: Counter, count: int) -> list[dict]:
    """A few prompts that differ from each other: the most typical first, then each time the
    one least like those already picked."""
    candidates = [i for i in items if i.get("positive")]
    if not candidates:
        return []
    sets = {id(i): phrases(i["positive"]) or set(i["positive"].lower().split()) for i in candidates}

    def distance(a: dict, b: dict) -> float:
        x, y = sets[id(a)], sets[id(b)]
        return 1 - len(x & y) / len(x | y) if x | y else 0.0

    picked = [max(candidates, key=lambda i: sum(frequency[p] for p in sets[id(i)]) / (len(sets[id(i)]) or 1))]
    while len(picked) < min(count, len(candidates)):
        best = max((i for i in candidates if i not in picked),
                   key=lambda i: min(distance(i, p) for p in picked))
        picked.append(best)
    return picked


def _font(size: int):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # an older Pillow
        return ImageFont.load_default()


def contact_sheets(tiles: list[tuple[str, Path]]) -> list[bytes]:
    """JPEG sheets of up to SHEET_COLUMNS x SHEET_ROWS thumbnails, each with its label (R1, R2...)."""
    per_sheet = SHEET_COLUMNS * SHEET_ROWS
    font = _font(28)
    sheets = []
    for start in range(0, len(tiles), per_sheet):
        chunk = tiles[start:start + per_sheet]
        rows = -(-len(chunk) // SHEET_COLUMNS)
        sheet = Image.new("RGB", (SHEET_COLUMNS * TILE, rows * TILE), (24, 22, 20))
        draw = ImageDraw.Draw(sheet)
        for index, (label, file) in enumerate(chunk):
            x, y = (index % SHEET_COLUMNS) * TILE, (index // SHEET_COLUMNS) * TILE
            try:
                with Image.open(file) as image:
                    image.draft("RGB", (TILE, TILE))  # JPEGs decode smaller, faster
                    thumb = ImageOps.contain(ImageOps.exif_transpose(image).convert("RGB"), (TILE - 8, TILE - 8))
                sheet.paste(thumb, (x + (TILE - thumb.width) // 2, y + (TILE - thumb.height) // 2))
            except Exception as e:
                logger.info("reference thumbnail failed file=%s error=%r", file, e)
            box = draw.textbbox((0, 0), label, font=font)
            draw.rectangle((x + 4, y + 4, x + 16 + box[2] - box[0], y + 14 + box[3] - box[1]), fill=(0, 0, 0))
            draw.text((x + 10 - box[0], y + 8 - box[1]), label, fill=(242, 165, 65), font=font)
        out = io.BytesIO()
        sheet.save(out, "JPEG", quality=85)
        sheets.append(out.getvalue())
    return sheets


def study(entry: dict, directory: Path, installed: dict | None = None) -> Study:
    """The study of an accepted folder (or a folder inside it) that goes with a message."""
    root = Path(entry["path"])
    if not root.is_dir():
        raise ReferenceFolderError(f"The folder {entry['name']} isn't at {entry['path']} any more.")
    files, capped = images_in(directory, root)
    where = entry["name"] if directory == root else f"{entry['name']} ({directory})"
    if not files:
        raise ReferenceFolderError(f"There are no images (PNG, JPEG or WebP) in {where}.")
    picked = _spread(files, MAX_STUDIED)
    items = [item for item in (_read(file, root) for file in picked) if item]
    if not items:
        raise ReferenceFolderError(f"None of the images in {where} could be read.")
    described = [i for i in items if "generator" in i]

    models, generators, shapes = Counter(), Counter(), Counter()
    steps, cfgs, samplers, schedulers = Counter(), Counter(), Counter(), Counter()
    loras: dict[str, list] = {}
    lora_images: Counter = Counter()
    positive, negative = Counter(), Counter()
    for item in items:
        generators[item.get("generator", "no metadata")] += 1
        if item.get("shape"):
            shapes[item["shape"]] += 1
        if "generator" not in item:
            continue
        for model in dict.fromkeys(item["models"]):
            models[model] += 1
        for name, weight in item["loras"]:
            loras.setdefault(name, [])
            if weight is not None:
                loras[name].append(weight)
        for name in dict.fromkeys(n for n, _ in item["loras"]):
            lora_images[name] += 1
        for counter, key in ((steps, "steps"), (cfgs, "cfg"), (samplers, "sampler"), (schedulers, "scheduler")):
            if item.get(key) not in (None, ""):
                counter[item[key] if isinstance(item[key], (int, float)) else str(item[key])] += 1
        positive.update(phrases(item["positive"]))
        negative.update(phrases(item["negative"]))

    total = len(items)
    head = (f'[Reference study: the user points you at "{entry["name"]}", a folder of images they picked '
            f"({directory}). {len(files)}{'+' if capped else ''} images")
    head += f"; {total} studied, spread evenly over the folder" if total < len(files) else ""
    head += f"; {len(described)} of them carry generation metadata (a prompt, models, settings).]"
    lines = [head, "", INSTRUCTIONS.format(name=entry["name"]), "", "## Recipes"]
    if models:
        lines.append("Base models (images): " + "; ".join(
            f"{_file_name(m)} ({n}{'; ' + note if (note := _installed(m, 'checkpoint', installed)) else ''})"
            for m, n in models.most_common(TOP_VALUES)))
    if lora_images:
        def lora_text(name: str, n: int) -> str:
            weights = loras.get(name) or []
            spread = ""
            if weights:
                low, high, mid = min(weights), max(weights), statistics.median(weights)
                spread = f"; weight {_fmt(mid)}" + (f", {_fmt(low)} to {_fmt(high)}" if low != high else "")
            note = _installed(name, "lora", installed)
            return f"{_file_name(name)} ({n}{spread}{'; ' + note if note else ''})"
        lines.append("LoRAs (images; median weight, range): " + "; ".join(
            lora_text(name, n) for name, n in lora_images.most_common(20)))
    for label, counter in (("Samplers", samplers), ("Schedulers", schedulers), ("Steps", steps), ("CFG", cfgs)):
        if counter:
            lines.append(f"{label}: {_tally(counter)}")
    if shapes:
        lines.append(f"Proportions: {_tally(shapes)}")
    lines.append(f"Made with: {_tally(generators)}")

    with_prompt = sum(1 for i in described if i["positive"])
    if positive:
        floor = max(2, round(0.05 * with_prompt))
        common = [(p, n) for p, n in positive.most_common(TOP_PHRASES) if n >= floor]
        if common:
            lines += ["", f"## Phrases that recur in the prompts (images, of {with_prompt} with a prompt)",
                      ", ".join(f'"{p}" ({n})' for p, n in common)]
        neg_common = [(p, n) for p, n in negative.most_common(TOP_NEGATIVE) if n >= floor]
        if neg_common:
            lines.append("Negative: " + ", ".join(f'"{p}" ({n})' for p, n in neg_common))

    samples = _pick_samples(described, positive, SAMPLE_PROMPTS)
    tiles_items = samples + [i for i in _spread([i for i in items if i not in samples],
                                                 MAX_SHEETS * SHEET_COLUMNS * SHEET_ROWS - len(samples))]
    tiles_items = tiles_items[:MAX_SHEETS * SHEET_COLUMNS * SHEET_ROWS]
    labels = {id(item): f"R{n}" for n, item in enumerate(tiles_items, 1)}
    if samples:
        lines += ["", "## Some whole prompts, chosen to differ from each other"]
        for item in samples:
            text = item["positive"]
            text = text if len(text) <= MAX_SAMPLE_CHARS else text[:MAX_SAMPLE_CHARS] + "..."
            lines.append(f"{labels[id(item)]} ({item['name']}; {_recipe(item)}):\nPOSITIVE: {text}")
            if item["negative"]:
                lines.append(f"NEGATIVE: {item['negative'][:500]}")
    sheets = contact_sheets([(labels[id(item)], item["file"]) for item in tiles_items])
    lines += ["", f"## Contact sheets: the {len(sheets)} image(s) attached to this message, in order",
              f"Thumbnails labelled R1 to R{len(tiles_items)}: the prompts above first, then a spread of the rest.",
              *(f"{labels[id(item)]} {item['name']} -- {_recipe(item)}" for item in tiles_items)]

    counts = {"name": entry["name"], "path": str(directory), "images": len(files), "capped": capped,
              "studied": total, "with_metadata": len(described), "sheets": len(sheets)}
    logger.info("reference study folder=%s images=%d studied=%d with_metadata=%d sheets=%d",
                entry["name"], len(files), total, len(described), len(sheets))
    return Study("\n".join(lines), sheets, counts)

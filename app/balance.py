"""Taking stock of a conversation (the chat's "Hacer balance").

The bot learns run by run: it sees the latest images only, its own reviews of older ones
aren't replayed to it, and a note it took after one image is rarely looked at again. Taking
stock gives it the whole conversation at once: every image the runs made, with the recipe it
was made with (base model, LoRAs and weights, sampler settings, size, seed, and the proposal
whose prompt it ran), the bot's review and the user's reaction, plus counts per ingredient.
The app builds it from what is stored -- the bot doesn't dig for it -- and it goes as the text
of a user-side message (stored on it, so later turns replay it unchanged and the prompt cache
keeps serving them), asking for what worked best and worst, and for the notes to be brought
in line with the evidence. The data alone (ledger) is also what an external agent gets.
"""
import re
from collections import Counter
from dataclasses import dataclass, field

from . import chat, conversations, prompts
from .comfy_graph import Graph, GraphError

# The latest runs, when a conversation has more: the text stays at a few thousand tokens.
MAX_RUNS = 120
# Per ingredient, the values with the most images (the rest counted in one line); LoRAs, which
# come several to an image, get more.
MAX_VALUES = 15
MAX_LORA_VALUES = 30
# "1:1 (Square), 1.0 MP" (runs from before) and "1:1 square, 1 MP" are the same proportions.
SHAPE = re.compile(r"^(\d+:\d+)\b.*?([\d.]+)\s*MP\b", re.IGNORECASE)
MODEL_EXTENSIONS = (".safetensors", ".ckpt", ".gguf", ".sft", ".pt", ".pth", ".bin")

INTRO = (
    "[The user asked you to take stock of this whole conversation (the chat's \"take stock\" button). "
    "Below, made by the app from what is stored: every image the runs made here, with the recipe it was "
    "made with, your review of it and the user's reaction to it, then the counts per ingredient.]"
)

INSTRUCTIONS = """Take stock, in the user's language:
- what worked best and what worked worst, and why -- each point with how many images back it ("3 of 4 liked at 0.6");
- the recipe that worked best: base model, LoRAs and weights, settings, and the proposal (by its number) whose prompt led there;
- what is still untested or unclear, and what you'd try next.
The user's judgment comes first: their reactions, and what they said about the images in the conversation itself. Your reviews come second. An image the user didn't react to isn't a verdict either way. A conclusion that rests on one or two images is a hunch: say so.
Then bring your notes in line with this evidence: rewrite the ones it corrects or sharpens, merge near-duplicates, forget the ones it contradicts, and add what it teaches that isn't noted yet (as "<model name>: <effect>"). Note a hunch only as one, if at all. Don't launch anything in this reply."""


@dataclass
class Ledger:
    text: str
    images: int = 0
    runs: int = 0
    liked: int = 0
    disliked: int = 0
    unrated: int = 0
    reviewed: int = 0
    flawed: int = 0
    rows: list[dict] = field(default_factory=list)

    @property
    def counts(self) -> dict:
        return {key: getattr(self, key) for key in
                ("images", "runs", "liked", "disliked", "unrated", "reviewed", "flawed")}


def _short(file: str) -> str:
    """"krea-2\\turbo_v1.0.safetensors" -> "krea-2\\turbo_v1.0"."""
    for extension in MODEL_EXTENSIONS:
        if file.lower().endswith(extension):
            return file[:-len(extension)]
    return file


def _lora(file: str) -> str:
    """A LoRA without its folder or extension: the base model already names the family."""
    return _short(file).replace("/", "\\").rsplit("\\", 1)[-1]


def _size(text) -> str | None:
    if not text:
        return None
    match = SHAPE.match(str(text))
    return f"{match.group(1)}, {float(match.group(2)):g} MP" if match else str(text)


def _squash(text: str | None) -> str:
    return " ".join((text or "").split()).lower()


def proposal_numbers(messages: list[dict]) -> dict[str, int]:
    """Each proposed prompt (squashed) and the number of the first block that proposed it, as
    the bot sees them numbered (chat.build_llm_messages)."""
    numbers: dict[str, int] = {}
    count = 0
    for message in messages:
        if message["role"] != "assistant":
            continue
        for step in message.get("steps") or [{"content": message.get("content") or ""}]:
            for match in chat.PROMPT_PATTERN.finditer(step.get("content") or ""):
                count += 1
                positive, _ = prompts.parse_response(match.group(2), True)
                numbers.setdefault(_squash(positive), count)
    return numbers


def _sent_info(sent: dict | None) -> tuple[dict, list[str]]:
    """What the workflow sent for a job says: generation_info() and its base models."""
    if not isinstance(sent, dict):
        return {}, []
    try:
        graph = Graph(sent)
        return graph.generation_info(), graph.base_models()
    except (GraphError, KeyError, TypeError, ValueError):
        return {}, []


def _rows(conversation: dict) -> tuple[list[dict], int]:
    """One row per image a run made, oldest first; and how many runs were left out (MAX_RUNS)."""
    messages = conversation["messages"]
    numbers = chat.image_numbers(messages)
    images = {img["id"]: img for m in messages for img in m.get("images") or []}
    proposals = proposal_numbers(messages)
    made = [r for r in conversation.get("runs") or [] if any(j.get("images") for j in r.get("jobs") or [])]
    left_out = max(0, len(made) - MAX_RUNS)
    rows = []
    for run in made[left_out:]:
        sent = conversations.run_workflows(conversation["id"], run["n"])
        for index, job in enumerate(run.get("jobs") or []):
            if not job.get("images"):
                continue
            info, base_models = _sent_info(sent[index] if index < len(sent) else None)
            settings = info.get("settings") or {}
            positive = (run.get("changes") or {}).get("positive") or info.get("positive")
            for image_id in job["images"]:
                if image_id not in numbers:
                    continue
                image = images.get(image_id) or {}
                reaction, review = image.get("reaction") or {}, image.get("review") or {}
                rows.append({
                    "number": numbers[image_id],
                    "run": run["n"],
                    "calibration": run.get("lora") if run.get("kind") == "calibration" else None,
                    "weight": job.get("calibration_weight"),
                    "workflow": run.get("workflow"),
                    "proposal": proposals.get(_squash(positive)),
                    "models": [_short(m) for m in run.get("base_models") or base_models],
                    "loras": [(_lora(lora["name"]), lora.get("strength")) for lora in job.get("loras") or []],
                    "sampler": "/".join(str(settings[k]) for k in ("sampler_name", "scheduler") if settings.get(k)),
                    "steps": settings.get("steps"),
                    "cfg": settings.get("cfg"),
                    "denoise": settings.get("denoise"),
                    "size": _size(run.get("size") or settings.get("size")),
                    "seed": job.get("seed") if job.get("seed") is not None else settings.get("seed"),
                    "rating": reaction.get("rating"),
                    "comment": " ".join((reaction.get("comment") or "").split()),
                    "verdict": review.get("verdict"),
                    "note": " ".join((review.get("note") or "").split()),
                })
    return rows, left_out


def _number(value) -> str:
    return f"{value:g}" if isinstance(value, (int, float)) else str(value)


def _what(row: dict) -> str:
    """What a run made: the workflow, and the proposal whose prompt it ran."""
    pieces = [f"calibration of {_lora(row['calibration'])}"] if row["calibration"] else []
    pieces.append(f"workflow {row['workflow']}")
    pieces.append(f"proposal {row['proposal']}" if row["proposal"] else "a prompt not among your proposals")
    return ", ".join(pieces)


def _settings(row: dict) -> str:
    """How it was made: base model, LoRAs, sampler settings, size."""
    pieces = []
    if row["models"]:
        pieces.append(", ".join(row["models"]))
    if row["loras"] and not row["calibration"]:
        pieces.append("LoRAs " + ", ".join(f"{name} {_number(w)}" for name, w in row["loras"]))
    elif not row["loras"]:
        pieces.append("no LoRAs")
    sampler = [row["sampler"]] if row["sampler"] else []
    if row["steps"] is not None:
        sampler.append(f"{_number(row['steps'])} steps")
    if row["cfg"] is not None:
        sampler.append(f"cfg {_number(row['cfg'])}")
    if isinstance(row["denoise"], (int, float)) and row["denoise"] != 1:
        sampler.append(f"denoise {_number(row['denoise'])}")
    if sampler:
        pieces.append(" ".join(sampler[:1]) + (", " + ", ".join(sampler[1:]) if sampler[1:] else ""))
    if row["size"]:
        pieces.append(str(row["size"]))
    return "; ".join(pieces)


def _verdicts(row: dict) -> str:
    user = "user " + {"up": "liked", "down": "disliked"}.get(row["rating"], "no reaction")
    if row["comment"]:
        user += f': "{row["comment"]}"'
    if row["verdict"]:
        mine = f"you: {row['verdict']}" + (f': "{row["note"]}"' if row["note"] else "")
    else:
        mine = "you: not reviewed"
    return f"{user}; {mine}"


def _image_line(row: dict) -> str:
    extra = []
    if row["weight"] is not None:
        extra.append(f"weight {_number(row['weight'])}")
    if row["seed"] is not None:
        extra.append(f"seed {row['seed']}")
    return f"- Image {row['number']}" + (f" ({', '.join(extra)})" if extra else "") + f": {_verdicts(row)}."


def _runs_section(rows: list[dict]) -> str:
    """Images grouped by run, the recipe once per run (it changes between runs, not within one,
    except a calibration's weight and each image's seed). Settings the same as the run before
    are said to be, not repeated."""
    lines, current, previous = [], None, None  # previous: (the run that spelled them out, settings)
    for row in rows:
        if row["run"] != current:
            current, settings = row["run"], _settings(row)
            if previous and settings == previous[1]:
                lines.append(f"Run {row['run']}: {_what(row)}; same settings as run {previous[0]}")
            else:
                lines.append(f"Run {row['run']}: {_what(row)}; {settings}")
                previous = (row["run"], settings)
        lines.append(_image_line(row))
    return "\n".join(lines)


INGREDIENTS = (
    ("Base model", lambda r: r["models"]),
    ("LoRA and weight", lambda r: [f"{name} at {_number(w)}" for name, w in r["loras"]]
     + ([f"{_lora(r['calibration'])} at {_number(r['weight'])}"] if r["calibration"] and r["weight"] is not None else [])),
    ("Sampler", lambda r: [r["sampler"]] if r["sampler"] else []),
    ("Steps", lambda r: [_number(r["steps"])] if r["steps"] is not None else []),
    ("CFG", lambda r: [_number(r["cfg"])] if r["cfg"] is not None else []),
    ("Size", lambda r: [str(r["size"])] if r["size"] else []),
    ("Workflow", lambda r: [r["workflow"]] if r["workflow"] else []),
    ("Proposal", lambda r: [f"proposal {r['proposal']}"] if r["proposal"] else []),
)


def _tally(rows: list[dict]) -> str:
    count = Counter()
    for row in rows:
        count["images"] += 1
        count[{"up": "liked", "down": "disliked"}.get(row["rating"], "unrated")] += 1
        if row["verdict"] == "flawed":
            count["flawed"] += 1
    text = f"{count['images']} image{'s' if count['images'] != 1 else ''}: {count['liked']} liked, " \
           f"{count['disliked']} disliked, {count['unrated']} unrated"
    if count["flawed"]:
        text += f"; {count['flawed']} flawed in your reviews"
    return text


def _ingredients_section(rows: list[dict]) -> str:
    sections = []
    for label, values_of in INGREDIENTS:
        groups: dict[str, list[dict]] = {}
        for row in rows:
            for value in dict.fromkeys(values_of(row)):
                groups.setdefault(value, []).append(row)
        if not groups:
            continue
        if len(groups) == 1 and len(next(iter(groups.values()))) == len(rows):
            sections.append(f"{label}: {next(iter(groups))} in every image.")
            continue
        lora = label == "LoRA and weight"
        ordered = sorted(groups.items(), key=lambda item: -len(item[1]))
        limit = MAX_LORA_VALUES if lora else MAX_VALUES
        shown, rest = ordered[:limit], ordered[limit:]
        if lora:  # read side by side: one LoRA at each weight
            shown.sort(key=lambda item: item[0].lower())
        lines = [f"- {value}: {_tally(group)}" for value, group in shown]
        if rest:
            lines.append(f"- {len(rest)} more, each in fewer images")
        sections.append(f"{label}:\n" + "\n".join(lines))
    return "\n\n".join(sections)


def ledger(conversation: dict) -> Ledger | None:
    """The evidence for taking stock of a conversation; None when its runs made no images yet."""
    rows, left_out = _rows(conversation)
    if not rows:
        return None
    rated = Counter(row["rating"] for row in rows)
    found = Ledger(
        text="",
        images=len(rows),
        runs=len({row["run"] for row in rows}),
        liked=rated["up"],
        disliked=rated["down"],
        unrated=len(rows) - rated["up"] - rated["down"],
        reviewed=sum(1 for row in rows if row["verdict"]),
        flawed=sum(1 for row in rows if row["verdict"] == "flawed"),
        rows=rows,
    )
    head = (f"{found.images} images from {found.runs} runs: the user liked {found.liked}, disliked {found.disliked} "
            f"and didn't react to {found.unrated}; you reviewed {found.reviewed} ({found.flawed} flawed).")
    if left_out:
        head += f" The first {left_out} runs that made images are left out; only the latest {MAX_RUNS} are here."
    found.text = "\n\n".join([
        head,
        f"## Every image, by run\n{_runs_section(rows)}",
        f"## Counts per ingredient\n{_ingredients_section(rows)}",
    ])
    return found


def request_text(found: Ledger) -> str:
    """The message that asks the bot to take stock: what it is, the evidence, what to do with it."""
    return f"{INTRO}\n\n{found.text}\n\n{INSTRUCTIONS}"


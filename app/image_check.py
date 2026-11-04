"""An independent check of a run's images, for Claude Code conversations.

The bot judges images it has just asked for, knowing what they should show, and a whole batch at
a time: it sees what it expected, and missed about half of what the user disliked (9 of 21 on
2026-11-04). So before the turn that first sees a run's images, a separate one-off request --
without the conversation, not told what was intended, told that most generated images have a
flaw -- inspects each at the size the model takes, with a retoucher's checklist. Its findings
reach the bot in the turn's context (context_text), which it weighs in its reviews and confirms
by zooming in (get_image with regions); they stay on each image (`check`), so a retry or a redo
doesn't pay for them again, and the chat shows them under each image.

One request per batch, up to MAX_IMAGES images: their tokens and a short answer, no session
re-read -- cheaper than any call of the turn itself. A conversation turns it off with
`image_check: false` ("Revisión independiente de las imágenes").
"""
import json
import logging
import re
from datetime import datetime, timezone

from . import chat, claude_code, conversations

logger = logging.getLogger("app.conversations")

MAX_IMAGES = 12
MAX_ISSUES = 6
VERDICTS = ("ok", "flawed")
SEVERITIES = ("minor", "major")

SYSTEM = """You inspect AI-generated images for visible defects, the way a demanding retoucher would before delivering them. You don't know what each image was meant to show, and you don't judge taste or whether it fits a request: only what is visibly wrong in it.

Go over each image region by region before deciding, and check:
- Hands: count the fingers of every visible hand; fingers fused, missing, extra, twisted or boneless; a hand merging with an object or another hand.
- Limbs and bodies: extra or missing arms or legs, joints bending the wrong way, impossible poses, wrong proportions, bodies merging with each other or with furniture, duplicated bodies or heads.
- Faces: eyes misaligned, of different sizes or looking different ways; pupils, teeth, ears, mouth; a melted or smeared face; a face where there shouldn't be one.
- Continuity: clothing, straps, jewelry or hair that break, change or vanish behind something; patterns that don't continue; objects passing through others.
- Background and objects: lines that should be straight and aren't, broken perspective, misshapen objects, garbled text or signs.
- Rendering: styles mixed within the image, smeared or plastic skin, oversharpening, noise, banding, burnt or washed-out colors, seams or tiling.

Most generated images have at least one flaw: look closely for it. But report only what you can actually see, never what is merely likely. "minor": noticeable on a close look; "major": spoils the image at first glance.

Answer with JSON only, one entry per image:
{"images": [{"image": <its number>, "verdict": "ok" or "flawed", "issues": [{"what": "...", "where": "...", "severity": "minor" or "major"}]}]}
"ok" means no issues. Write "what" and "where" in Spanish, a few words each (e.g. "seis dedos en la mano izquierda", "abajo a la izquierda")."""

CONTEXT_HEADING = (
    "## Independent Check Of The New Images\n"
    "A separate inspection looked at each new image at full size for visible defects, without this conversation and "
    "without knowing what was asked. Weigh it in your reviews -- it can be wrong, and it doesn't judge whether an image "
    "is what was asked -- and confirm what matters by zooming in (get_image with regions)."
)


def enabled(conversation: dict) -> bool:
    return claude_code.engine(conversation) and conversation.get("image_check", True) is not False


def _new_images(conversation: dict) -> list[dict]:
    """The run images the bot hasn't spoken since (the ones its turn reviews), but calibrations'."""
    messages = conversation["messages"]
    last_answer = max((i for i, m in enumerate(messages) if m["role"] == "assistant"), default=-1)
    runs = {r["n"]: r for r in conversation.get("runs") or []}
    images = []
    for message in messages[last_answer + 1:]:
        if message["role"] != "results" or (runs.get(message.get("run")) or {}).get("kind") == "calibration":
            continue
        images += message.get("images") or []
    return images


def _parsed(text: str, numbers: set[int]) -> dict[int, dict]:
    """The inspector's answer, kept to what holds: numbers asked about, known verdicts, short issues."""
    match = re.search(r"\{.*\}", text or "", re.DOTALL)
    try:
        data = json.loads(match.group(0)) if match else {}
    except ValueError:
        data = {}
    found = {}
    for entry in data.get("images") or [] if isinstance(data, dict) else []:
        if not isinstance(entry, dict) or entry.get("image") not in numbers:
            continue
        issues = []
        for issue in entry.get("issues") or []:
            if isinstance(issue, dict) and str(issue.get("what") or "").strip():
                severity = issue.get("severity") if issue.get("severity") in SEVERITIES else "minor"
                issues.append({"what": " ".join(str(issue["what"]).split())[:120],
                               "where": " ".join(str(issue.get("where") or "").split())[:60], "severity": severity})
        verdict = "flawed" if issues else "ok"
        found[entry["image"]] = {"verdict": verdict, "issues": issues[:MAX_ISSUES]}
    return found


def check(conversation: dict, load_image) -> int:
    """Inspect the new run images not inspected yet (stored on each as `check`); how many were.
    A failure is logged and skipped: the turn goes on without it."""
    if not enabled(conversation):
        return 0
    numbers = chat.image_numbers(conversation["messages"])
    pending = [i for i in _new_images(conversation) if not i.get("check") and i["id"] in numbers][:MAX_IMAGES]
    blocks, asked = [], []
    for image in pending:
        uri = load_image(image)
        if not uri:
            continue
        header, _, data = uri.partition(",")
        blocks.append({"type": "image", "source": {"type": "base64", "media_type": header[5:].split(";")[0],
                                                   "data": data}})
        asked.append(numbers[image["id"]])
    if not asked:
        return 0
    listed = ", ".join(f"image {n}" for n in asked)
    model = claude_code.model_for(conversation)
    try:
        text, info = claude_code.ask(SYSTEM, [{"type": "text", "text": f"Inspect these images, in this order: {listed}."},
                                              *blocks], model,
                                     {"mode": "image_check", "conversation_id": conversation["id"]})
    except claude_code.AskError as e:
        logger.warning("image check failed conversation_id=%s: %s", conversation["id"], e)
        return 0
    found = _parsed(text, set(asked))
    now = datetime.now(timezone.utc).isoformat()
    by_number = {n: image_id for image_id, n in numbers.items()}
    checks = {by_number[n]: {**found.get(n, {"verdict": "unknown", "issues": []}), "model": model, "at": now}
              for n in asked}
    conversations.set_image_checks(conversation["id"], checks)
    logger.info("image check conversation_id=%s images=%d flawed=%d cost=%s", conversation["id"], len(asked),
                sum(1 for c in checks.values() if c["verdict"] == "flawed"), info.get("estimated_cost"))
    return len(asked)


def context_text(conversation: dict) -> str | None:
    """What the inspection found in the images this turn reviews, for the turn's context."""
    numbers = chat.image_numbers(conversation["messages"])
    lines = []
    for image in _new_images(conversation):
        found = image.get("check") or {}
        if found.get("verdict") not in VERDICTS or image["id"] not in numbers:
            continue
        if found["verdict"] == "ok":
            lines.append(f"- Image {numbers[image['id']]}: nothing found.")
        else:
            issues = "; ".join(f"{i['what']} ({i['where'] + ', ' if i['where'] else ''}{i['severity']})"
                               for i in found["issues"])
            lines.append(f"- Image {numbers[image['id']]}: {issues}.")
    return "\n".join([CONTEXT_HEADING, *lines]) if lines else None

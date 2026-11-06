"""The away mode ("Modo autónomo", 2026-11-05): while the user is away -- the night, a long absence --
the bot works on its own, in a conversation of its own, making what it thinks they would love and
what it is curious to try, on what would otherwise go unused: the GPU while nothing else wants it,
and the Claude Code subscription's allowance.

The user picks what the night is for (`focus`): their favorites (copied in, for variations, finishing,
calibrating their LoRAs), surprising them ("explore": original images only loosely based on their taste,
away from what they usually make, starting from sparks the app draws at random -- sparks.py: an LLM left
to itself keeps arriving at the same few ideas), or half and half ("mix", the default).

It must never get in the way of the user's own use (the owner's rules):
- ComfyUI: everything it launches goes at comfy.PRIORITY_AWAY ("idle"), the one level the gateway
  still stops halfway: it runs only while nothing else wants the GPU, and gives way at once.
- Claude Code's 5-hour window, which opens with its first use and renews 5 hours later. A turn runs
  in a window that renews before the user is back -- all of it may be used, up to FULL_SHARE -- or
  in the one still open when they are back, while it has used at most OVERLAP_SHARE of it: they find
  75% of that one left, at least. With no window open, a turn opens one that ends 5 hours later.
  Each turn's share is estimated from what the away turns took so far (TURN_SHARE at least), and
  checked before every turn (`outlook`); out of room until a window renews, it waits for it
  (retry_later, as after a limit). The weekly window has no ceiling of its own (the owner's call):
  it is only never run out (OTHER_SHARE).
- The user: anything they do that makes the bot work -- a message, a retry, stopping it -- in any
  conversation ends it (user_active): they are back.

It ends at `until`, when the user said they'd be back (a timer, kept across restarts: resume), or
before, when the allowance runs out. Its last turn -- FINAL_SECONDS before `until`, or when the
allowance won't cover another -- sums up the night for the user. Runs still under way then go on at
idle, and their images land in the chat without a comment (the conversation's own comments are off).
"""
import logging
import random
import threading
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from . import catalog, chat, claude_code, comfy, conversations, families, retry_later, sparks, storage
from .config import get_settings

logger = logging.getLogger(__name__)

KIND = "away"
WINDOW_SECONDS = 5 * 3600
FULL_SHARE = 0.95  # a window that renews before the user is back: nearly all of it, not into a refusal
OVERLAP_SHARE = 0.25  # the one still open when they're back: they find 75% of it left
OTHER_SHARE = 0.95  # the weekly window: no ceiling but not running it out
TURN_SHARE = 0.05  # a turn's share of the 5-hour window, at least (the bot's turns: median 2%, p90 5%)
RECENT_TURNS = 3  # the estimate: the largest of the latest turns' shares
FINAL_SECONDS = 20 * 60  # the last turn: what it launched after this couldn't be reviewed before the end
MIN_SECONDS = 30 * 60
MAX_SECONDS = 24 * 3600
MAX_IMAGES = 400
# What the night is for (the user picks it when starting): their favorites, surprising them, or half and half.
# Per focus, how many favorites are copied in for the bot to start from, and how many sparks are drawn.
FOCUSES = ("mix", "explore", "favorites")
FAVORITES = {"favorites": 8, "mix": 4, "explore": 0}
SPARKS = {"favorites": 0, "mix": 3, "explore": 5}
RECENT_NIGHTS = 3  # the sparks of these latest nights aren't drawn again (while there are others)
THEMES = 10  # the latest conversations' titles: what the user usually makes, for exploring to step away from
UNUSED = 8  # installed models and LoRAs never used here, a random few, for exploring
BASE_FOLDERS = ("checkpoints", "diffusion_models")

_timer: threading.Timer | None = None
_guard = threading.Lock()


class AwayError(ValueError):
    pass


@dataclass
class Outlook:
    """What the next turn may do. action: "go" (a turn with tools), "last" (the last turn: it sums
    up), "wait" (until `at`, when the 5-hour window renews) or "end" (reason "until" or "limit")."""
    action: str
    reason: str = ""
    at: float | None = None
    turns: int = 0  # turns the allowance still covers in this window, this one included
    resets_at: float | None = None
    capped: bool = False  # this window is still open when the user is back: OVERLAP_SHARE of it at most
    renews: bool = False  # it renews in time for more turns before the end


def is_away(working: dict | None) -> bool:
    return bool(working) and working.get("active") and working.get("kind") == KIND


def outlook(working: dict, limits: dict | None, now: float, turn_share: float = TURN_SHARE) -> Outlook:
    """Whether a turn may run now, under the owner's rules (see the module's docstring), from the
    subscription's usage as the latest turn reported it (claude_code.limits)."""
    until = working["until"]
    if now >= until:
        return Outlook("end", reason="until")
    final_from = until - FINAL_SECONDS
    windows = (limits or {}).get("windows") or {}
    for name, window in windows.items():
        if name == "five_hour" or not isinstance(window, dict):
            continue
        resets = window.get("resets_at")
        if not (isinstance(resets, (int, float)) and resets > now):
            continue
        if (window.get("used") or 0) + turn_share > OTHER_SHARE or window.get("status") == "rejected":
            return Outlook("wait", at=resets) if resets < final_from else Outlook("end", reason="limit")
    five = windows.get("five_hour") or {}
    resets = five.get("resets_at")
    if isinstance(resets, (int, float)) and resets > now:
        used, open_ = five.get("used") or 0.0, True
    else:
        used, resets, open_ = 0.0, now + WINDOW_SECONDS, False  # a turn now opens a new window
    capped = resets > until
    room = (OVERLAP_SHARE if capped else FULL_SHARE) - used
    renews = open_ and resets < final_from
    if room < turn_share:
        return Outlook("wait", at=resets) if renews else Outlook("end", reason="limit")
    turns = int(room / turn_share + 1e-9)
    if now >= final_from:
        return Outlook("last", reason="until", turns=turns, resets_at=resets, capped=capped)
    if turns < 2 and not renews:
        return Outlook("last", reason="limit", turns=turns, resets_at=resets, capped=capped)
    return Outlook("go", turns=turns, resets_at=resets, capped=capped, renews=renews)


def turn_share(conversation: dict) -> float:
    """A turn's share of the 5-hour window, as the conversation's turns went: the rise from one reply
    to the next within the same window (which counts anything else that used it meanwhile too)."""
    seen = []
    for message in conversation["messages"]:
        info = ((message.get("claude_code") or {}).get("rate_limit") or {})
        window = (info.get("unifiedWindows") or {}).get("five_hour")
        if message["role"] == "assistant" and isinstance(window, dict) \
                and isinstance(window.get("utilization"), (int, float)):
            seen.append((window.get("resetsAt"), window["utilization"]))
    steps = [after - before for (window, before), (same, after) in zip(seen, seen[1:])
             if window == same and after >= before]
    return max([TURN_SHARE, *steps[-RECENT_TURNS:]])


# --- what the bot is told ---------------------------------------------------------------------------

DAYS = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")
MONTHS = ("January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
          "November", "December")
DIAS = ("lun", "mar", "mié", "jue", "vie", "sáb", "dom")
MESES = ("ene", "feb", "mar", "abr", "may", "jun", "jul", "ago", "sep", "oct", "nov", "dic")


def _clock(ts: float) -> str:
    return datetime.fromtimestamp(ts).strftime("%H:%M")  # the PC's time: the app's zone


def _when(ts: float) -> str:
    moment = datetime.fromtimestamp(ts)
    return f"{moment:%H:%M} on {DAYS[moment.weekday()]} {moment.day} {MONTHS[moment.month - 1]}"


def _span(seconds: float) -> str:
    minutes = max(0, round(seconds / 60))
    return f"{minutes // 60} h {minutes % 60:02d} min" if minutes >= 60 else f"{minutes} min"


BRIEF = """[Away mode] The user is away until {until} and left you working on your own until then, in this conversation of yours, with GPU time and Claude Code allowance that would otherwise go unused. {focus}

How it works:
- In rounds: launch a batch, look at its images when they come back, launch the next. Everything you launch runs at idle priority, only while ComfyUI has nothing else to do, so it never gets in anyone's way (a job may be stopped and redone when something else comes).
- Every turn tells you the time and how much Claude Code allowance is left. Each turn uses some, the images you look at most of all: batches of 4-8 images, zooms only where they matter, no lookups the context already answers.
- The allowance you're shown is only what is spare: what the user will need when they're back is already set aside, so don't save any of it for them. Don't plan your own ending either: when the time or the allowance is about to run out, the app tells you the turn is your last, and you sum up the night then. Until then, keep launching -- the sparks, your own ideas, what worked tonight taken further; stop earlier only if there is truly nothing left worth trying.
- The user isn't here to confirm anything: note facts you saw about models, LoRAs and workflows, with how many images back them; a hunch only as one; nothing about their taste until they have reacted. Review every image (<review>), but rate none for them: <rating> is for their own words.

{sections}"""

FOCUS = {
    "favorites": "Make what you think they would love, starting from their favorites, and what you are curious to try.",
    "explore": ("Tonight is for surprising them: original images they wouldn't have thought to ask for, only loosely "
                "based on their taste -- not variations of what they have made, but new subjects, settings, styles "
                "and models."),
    "mix": ("Split the night: about half the rounds on their favorites (variations of them), half on surprising "
            "them -- original images they wouldn't have thought to ask for, only loosely based on their taste."),
}

FAVORITES_SECTION = """Their favorites -- where to start is your call:
- Variations of them (image:N with changes -- other seeds, proportions, situations, settings, light). Not finishing them: that belongs to their own conversation, where the user works on them.
- Calibrating the LoRAs they use that were never calibrated{loras}.
- Something of your own you think they would like, from their notes and what they loved.
{favorites}"""

EXPLORE_SECTION = """Surprising them:
- Start from the sparks below, which the app drew at random for tonight: each is a nudge, not an order. Read it through their taste, change or drop parts, combine two -- but keep what makes it unexpected. Avoid the obvious take, and their usual ground unless you give it a twist.
- Several ideas, each tried first with a small batch (3-4 images: different seeds and proportions); go further with those that work, drop the rest without regret. Each idea may call for its own model, LoRAs and workflow: try some they rarely use.
- Their taste, loosely: {taste}
- What they usually make (recent conversations): {themes}.
- Installed and never used here, a random few: {unused}.
Sparks for tonight:
{sparks}"""


def favorites_text(favorites: list[dict]) -> str:
    if not favorites:
        return "They haven't marked any favorite yet: go by their notes and the models' stats."
    lines = [f"Images #1-#{len(favorites)} here are their favorites, each with its recipe:"]
    for number, item in enumerate(favorites, 1):
        what = "a favorite (loved)" if item["rating"] == "love" else "liked"
        line = (f"- #{number}: {what}, from the conversation \"{item['title']}\", run with {item['workflow']}"
                f"; {'a finished image' if item['finished'] else 'a draft'}.")
        if item.get("comment"):
            line += f' They said: "{item["comment"]}"'
        lines.append(line)
    return "\n".join(lines)


def brief_text(until: float, focus: str, favorites: list[dict], loras: list[str],
               explore: dict | None = None) -> str:
    """The away conversation's first message. explore: what surprising the user starts from -- sparks,
    taste, themes, unused (the app's, see start)."""
    sections = []
    if focus != "explore":
        sections.append(FAVORITES_SECTION.format(
            favorites=favorites_text(favorites),
            loras=f": {', '.join(loras)}" if loras else " (none of them is uncalibrated)"))
    if focus != "favorites":
        explore = explore or {}
        sections.append(EXPLORE_SECTION.format(
            taste=explore.get("taste") or "nothing marked yet: go by their notes.",
            themes=", ".join(f'"{t}"' for t in explore.get("themes") or []) or "nothing yet",
            unused=", ".join(explore.get("unused") or []) or "none",
            sparks="\n".join(f"{n}. {'; '.join(spark)}" for n, spark in enumerate(explore.get("sparks") or [], 1))))
    return BRIEF.format(until=_when(until), focus=FOCUS[focus], sections="\n\n".join(sections))


def context_text(working: dict, seen: Outlook, now: float, first: bool) -> str:
    """The turn's context while it works away: the time and allowance left, and what to do now."""
    until = working["until"]
    lines = [f"## Away Mode\nThe user is back at {_when(until)}; it's {_clock(now)} now ({_span(until - now)} left). "
             f"{working['images']} images made so far tonight."]
    if seen.capped:
        lines.append(f"Claude Code: this 5-hour window is still open when the user is back, so you may use only a "
                     f"quarter of it: about {seen.turns} turns left, this one included.")
    elif seen.renews:
        lines.append(f"Claude Code: about {seen.turns} turns left in this 5-hour window, this one included; it renews "
                     f"at {_clock(seen.resets_at)}, with more.")
    else:
        lines.append(f"Claude Code: about {seen.turns} turns left, this one included.")
    if seen.action != "last":
        lines.append("That is only what's spare -- the user's share is set aside already -- so use it; the app will "
                     "tell you when a turn is your last.")
    focus = working.get("focus") or "favorites"
    if focus == "mix":
        lines.append("Tonight's split: about half the rounds on their favorites, half on surprising them.")
    elif focus == "explore":
        lines.append("Tonight is for surprising them: new ideas, from the sparks.")
    if seen.action == "last":
        lines.append("This is your last turn. Sum up the night for the user: what you tried and why, the best images "
                     "(#N), what you learned, what you'd try next"
                     + ("; each new idea apart -- what it was, and its best image or that it didn't work"
                        if focus != "favorites" else "")
                     + ". If you like, end it by launching one last batch for them to find (say so): its images won't "
                     "come back to you.")
        return "\n".join(lines)
    lines.append(f"Batches should be back by {_clock(until - FINAL_SECONDS)} for you to review them: size them by the "
                 "time per image and the queue.")
    lines.append("Start: say what you'll try tonight and why, and launch the first round in this same reply."
                 if first else
                 "Review these images, decide the next round (focus or explore, and say which) and launch it in this "
                 "same reply, without asking. Keep going until the app gives you your last turn: stop earlier only "
                 "if nothing is left worth trying.")
    return "\n".join(lines)


# --- starting and stopping -----------------------------------------------------------------------------

def _pointer() -> Path:
    return Path(get_settings().data_dir, "away.json")


def current() -> str | None:
    """The conversation working away right now, if any."""
    found = storage.read_json(str(_pointer()), default=None) or {}
    conversation_id = found.get("conversation_id")
    if conversation_id and is_away(conversations.autonomy(conversations.get(conversation_id) or {})):
        return conversation_id
    return None


def _scan() -> tuple[list[dict], list[str], list[str]]:
    """One pass over every conversation: the user's favorites among the runs' images (loved first, then
    liked, newest first within each; with where they come from; reactions already sent only), the titles of their latest conversations
    (what they usually make; away ones aside), and the sparks of the latest away nights."""
    found, titled, nights = [], [], []
    root = Path(get_settings().data_dir, "conversations")
    for path in root.glob("*.json") if root.is_dir() else []:
        if not conversations.is_valid_id(path.stem):
            continue
        conversation = storage.read_json(str(path), default=None) or {}
        messages = conversation.get("messages") or []
        brief = (messages[0].get("away") if messages else None) or None
        if brief:
            nights.append((conversation.get("created_at") or "", [card for spark in brief.get("sparks") or []
                                                                   for card in spark]))
        elif conversation.get("title"):
            titled.append((conversation.get("updated_at") or "", conversation["title"]))
        numbers = chat.image_numbers(messages)
        runs = {run["n"]: run for run in conversation.get("runs") or []}
        finished = {run.get("finish_of") for run in runs.values() if run.get("kind") == "finish"}
        for message in messages:
            if message["role"] != "results":
                continue
            run = runs.get(message.get("run")) or {}
            for image in message.get("images") or []:
                reaction = image.get("reaction") or {}
                # One not sent yet is the user's next message there, for that conversation's bot to act on.
                if reaction.get("rating") not in ("love", "up") or not reaction.get("sent"):
                    continue
                job = next((j for j in run.get("jobs") or [] if image["id"] in (j.get("images") or [])), {})
                found.append({
                    "conversation_id": conversation["id"], "image_id": image["id"], "rating": reaction["rating"],
                    "at": reaction.get("at") or message.get("timestamp") or "",
                    "title": conversations.display_title(conversation) or "untitled",
                    "workflow": run.get("workflow") or "a workflow", "comment": (reaction.get("comment") or "").strip(),
                    "finished": run.get("kind") == "finish" or numbers.get(image["id"]) in finished,
                    "family_id": conversation.get("family_id"),
                    "loras": [lora["name"] for lora in job.get("loras") or [] if lora.get("name")],
                })
    found.sort(key=lambda item: item["at"], reverse=True)
    found.sort(key=lambda item: item["rating"] != "love")  # stable: loved first, each group newest first
    themes = list(dict.fromkeys(title for _, title in sorted(titled, reverse=True)))[:THEMES]
    recent = [card for _, cards in sorted(nights, reverse=True)[:RECENT_NIGHTS] for card in cards]
    return found, themes, recent


REASON_WORDS = {"hands": "hands", "face": "faces", "anatomy": "anatomy", "style": "style",
                "composition": "composition", "not_asked": "not what they asked for"}


def _taste(stats: dict, entries: dict) -> str:
    """Their taste in a line, from the catalog's stats: what they liked most, and what puts them off."""
    def liked(folders):
        rows = sorted(((record.get("liked", 0), record.get("loved", 0), file) for file, record in stats.items()
                       if record.get("liked") and (entries.get(file) or {}).get("folder") in folders), reverse=True)
        return ", ".join(f"{catalog.short_name(file)} ({count} liked"
                         f"{f', {loved} favorite' + ('s' if loved > 1 else '') if loved else ''})"
                         for count, loved, file in rows[:4])

    reasons: Counter = Counter()
    for file, record in stats.items():
        if (entries.get(file) or {}).get("folder") in BASE_FOLDERS:  # an image counts once: by its base model
            reasons.update(record.get("dislike_reasons") or {})
    parts = []
    if models := liked(BASE_FOLDERS):
        parts.append(f"the base models they liked most: {models}")
    if loras := liked(("loras",)):
        parts.append(f"LoRAs: {loras}")
    if reasons:
        parts.append("what puts them off most: " + ", ".join(f"{REASON_WORDS.get(r, r)} ({n})"
                                                             for r, n in reasons.most_common(3)))
    return "; ".join(parts) + "." if parts else ""


def _unused(stats: dict, entries: dict, rng: random.Random | None = None) -> list[str]:
    """A random few of the installed models and LoRAs no run here has used yet."""
    files = sorted(file for file, entry in entries.items()
                   if entry.get("folder") in (*BASE_FOLDERS, "loras") and not (stats.get(file) or {}).get("images"))
    picked = (rng or random.Random()).sample(files, min(UNUSED, len(files)))
    return [catalog.short_name(file) for file in sorted(picked)]


def _copy_image(conversation_id: str, item: dict) -> dict | None:
    """A favorite, into the away conversation as an image the user shared: with its workflow, so the bot
    can run it (image:N) or finish it."""
    found = conversations.image_file(item["conversation_id"], item["image_id"])
    if found is None:
        return None
    path, content_type = found
    image = conversations.save_image(conversation_id, Path(path).read_bytes(), content_type)
    workflow = conversations.image_workflow(item["conversation_id"], item["image_id"])
    if workflow and conversations.image_workflow(conversation_id, image["id"]) is None:
        storage.write_json(f"{get_settings().data_dir}/conversations/{conversation_id}/{image['id']}.workflow.json",
                           workflow)
    return image


def _title(now: float) -> str:
    moment = datetime.fromtimestamp(now)
    return f"Modo autónomo · {DIAS[moment.weekday()]} {moment.day} {MESES[moment.month - 1]}, {moment:%H:%M}"


def start(until: float, instructions: str = "", claude_model: str = "", claude_effort: str = "",
          family_id: str | None = None, focus: str = "mix", now: float | None = None) -> dict:
    """Open the away conversation and set the bot to work in it until `until` (epoch seconds), on `focus`:
    their favorites, surprising them ("explore"), or half and half ("mix"). Its first message is the
    app's: what the mode is, the favorites copied in, and for exploring the sparks drawn for the night,
    their taste in a line, what they usually make and a few models never used -- then their instructions."""
    now = time.time() if now is None else now
    if focus not in FOCUSES:
        raise AwayError(f"Unknown focus: {focus}")
    if comfy.client() is None:
        raise AwayError("ComfyUI isn't configured: there's nothing to work with")
    if not claude_code.available():
        raise AwayError("The away mode needs Claude Code, which isn't installed here (CLAUDE_CODE_COMMAND)")
    if not now + MIN_SECONDS <= until <= now + MAX_SECONDS:
        raise AwayError("The time you're back must be between half an hour and a day from now")
    running = current()
    if running:
        raise AwayError(f'The bot is already working on its own in "{conversations.display_title(conversations.get(running))}"')
    limits = claude_code.limits()
    if outlook({"until": until}, limits, now).action == "end":
        five = ((limits or {}).get("windows") or {}).get("five_hour") or {}
        raise AwayError(f"Claude Code leaves no room before you're back: its 5-hour window is at "
                        f"{round((five.get('used') or 0) * 100)}% and doesn't renew until {_clock(five['resets_at'])}"
                        if five.get("resets_at") else "Claude Code leaves no room before you're back")
    found, themes, recent = _scan()
    family = families.get_family(family_id or "") if family_id else None
    if family is None:
        family = next((families.get_family(f["family_id"]) for f in found if f.get("family_id")), None) \
            or (families.list_families() or [None])[0]
    if family is None:
        raise AwayError("There's no model family to start with")
    settings = get_settings()
    conversation = conversations.create(
        family["id"], settings.default_llm_model, 0.7, auto_comment=False, engine=claude_code.ENGINE,
        claude_model=claude_model if claude_model in claude_code.MODELS else settings.claude_code_model,
        claude_effort=claude_effort, image_check=False)
    conversation_id = conversation["id"]
    conversations.update_settings(conversation_id, {"title": _title(now)})
    copied = [(item, image) for item, image in ((item, _copy_image(conversation_id, item))
                                                for item in found[:FAVORITES[focus]]) if image]
    favorites, images = [item for item, _ in copied], [image for _, image in copied]
    stats = catalog.stats()
    loras = list(dict.fromkeys(name for item in favorites for name in item["loras"]
                               if not (stats.get(name) or {}).get("calibration")))
    explore = None
    if SPARKS[focus]:
        entries = catalog.load()["entries"] or {}
        explore = {"sparks": sparks.draw(SPARKS[focus], avoid=recent), "taste": _taste(stats, entries),
                   "themes": themes, "unused": _unused(stats, entries)}
    instructions = " ".join((instructions or "").split())
    conversations.append_message(conversation_id, {
        "role": "user", "content": instructions, "images": images,
        "away": {"until": until, "favorites": len(images), "focus": focus,
                 "sparks": explore["sparks"] if explore else []},
        "away_brief": brief_text(until, focus, favorites, [catalog.short_name(n) for n in loras], explore),
    })
    conversations.start_away(conversation_id, until, instructions, MAX_IMAGES, focus=focus)
    storage.write_json(str(_pointer()), {"conversation_id": conversation_id, "until": until})
    _schedule_end(conversation_id, until)
    logger.info("away mode started conversation_id=%s until=%s focus=%s favorites=%d sparks=%d", conversation_id,
                _clock(until), focus, len(images), SPARKS[focus])
    return conversations.get(conversation_id)


def stop(conversation_id: str, reason: str) -> dict | None:
    """End it: the bot launches nothing more and doesn't come back by itself. What it launched goes on
    at idle and lands in the chat without a comment."""
    ended = conversations.end_autonomy(conversation_id, reason)
    retry_later.cancel(conversation_id)
    if (conversations.get(conversation_id) or {}).get("retry_at"):
        conversations.set_retry(conversation_id, None)
    found = storage.read_json(str(_pointer()), default=None) or {}
    if found.get("conversation_id") == conversation_id:
        storage.remove(str(_pointer()))
    _cancel_end()
    if ended:
        logger.info("away mode ended conversation_id=%s reason=%s images=%s", conversation_id, reason,
                    ended.get("images"))
    return ended


def user_active() -> None:
    """The user did something that makes the bot work: they are back, and the away mode ends."""
    conversation_id = current()
    if conversation_id:
        stop(conversation_id, "user")


def _schedule_end(conversation_id: str, until: float) -> None:
    global _timer
    timer = threading.Timer(max(0.0, until - time.time()), _end_now, args=(conversation_id,))
    timer.daemon = True
    with _guard:
        previous, _timer = _timer, timer
    if previous is not None:
        previous.cancel()
    timer.start()


def _cancel_end() -> None:
    global _timer
    with _guard:
        previous, _timer = _timer, None
    if previous is not None:
        previous.cancel()


def _end_now(conversation_id: str) -> None:
    if is_away(conversations.autonomy(conversations.get(conversation_id) or {})):
        stop(conversation_id, "until")


def resume() -> None:
    """After a restart: the end of the one under way, scheduled again (or ended, if its time has passed)."""
    conversation_id = current()
    if not conversation_id:
        return
    until = conversations.autonomy(conversations.get(conversation_id))["until"]
    if until <= time.time():
        stop(conversation_id, "until")
    else:
        _schedule_end(conversation_id, until)

"""The Conversar bot's notes about the user, carried into conversations -- each one where it applies.

A note has a scope:
- "general": the user's tastes and way of working, true everywhere: sent in every turn;
- "family": what works or fails with a model family (`family`: its folder, "krea-2"): sent when
  the conversation's family writes for that folder (families.covers_folder);
- "topic": a character, a series, a project, a theme (`topic`: "Kim"): sent when the topic is
  active -- the conversation declares it (its `topics`, the bot's <topics>), or its name or one of
  its keywords appears in the latest messages or the title (like a lorebook's entries).
A note without a scope is general. Each turn sends what applies, within NOTES_BUDGET characters --
general first, then the family's, then the topics' (`notes_for_turn`), each group's most recently
confirmed first -- and names what was left out, so the bot knows it exists. `confirmed_at` and
`confirmations` count the times the bot found a note right again (<confirm id="n"/>), or rewrote it.
Topics' keywords (besides their name) are kept apart, in topics.json, edited in Ajustes.
"""
import re
import unicodedata
import uuid
from datetime import datetime, timezone

from . import storage
from .config import get_settings

SCOPES = ("general", "family", "topic")
NOTES_BUDGET = 4800  # characters of notes per turn: about 1,200 tokens
SCAN_MESSAGES = 6  # the latest messages a topic's name or keywords are looked for in
MAX_TOPIC_LENGTH = 40
MAX_TOPICS = 5  # per conversation


def _default_path() -> str:
    return f"{get_settings().data_dir}/memory.json"


def _topics_path() -> str:
    return f"{get_settings().data_dir}/topics.json"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def list_notes(path: str | None = None) -> list[dict]:
    return storage.read_json(path or _default_path(), default=[])


def _fold(text: str) -> str:
    """Lowercase without accents: "Doméstico" and "domestico" are the same word."""
    text = unicodedata.normalize("NFKD", text or "")
    return "".join(c for c in text if not unicodedata.combining(c)).lower()


def clean_scope(scope: str | None = None, family: str | None = None, topic: str | None = None,
                path: str | None = None) -> dict:
    """{"scope", and "family" or "topic"}: a family as its folder ("Krea 2" or "krea_2" -> "krea-2"),
    a topic as the notes already spell it ("kim" -> "Kim") or as given."""
    family = re.sub(r"[\s_]+", "-", (family or "").strip().lower()).strip("-")
    topic = " ".join((topic or "").split())[:MAX_TOPIC_LENGTH]
    if family and scope in (None, "family"):
        return {"scope": "family", "family": family}
    if topic and scope in (None, "topic"):
        known = {_fold(name): name for name in topic_names(path)}
        return {"scope": "topic", "topic": known.get(_fold(topic), topic)}
    return {"scope": "general"}


def _set_scope(note: dict, scope: dict) -> None:
    for key in ("scope", "family", "topic"):
        note.pop(key, None)
    note.update(scope)


def create_note(text: str, conversation_id: str | None = None, path: str | None = None,
                scope: dict | None = None) -> dict:
    resolved_path = path or _default_path()
    notes = list_notes(resolved_path)
    now = _now()
    note = {
        "id": str(uuid.uuid4()),
        "text": text,
        "created_at": now,
        "conversation_id": conversation_id,
        "confirmed_at": now,
        "confirmations": 0,
    }
    _set_scope(note, scope or {"scope": "general"})
    notes.append(note)
    storage.write_json(resolved_path, notes)
    return note


def update_note(note_id: str, text: str, path: str | None = None, scope: dict | None = None) -> dict | None:
    """Rewrite a note (and, given scope, move it): rewriting it is finding it right, updated."""
    resolved_path = path or _default_path()
    notes = list_notes(resolved_path)
    for note in notes:
        if note["id"] == note_id:
            note["text"] = text
            note["confirmed_at"] = _now()
            if scope is not None:
                _set_scope(note, scope)
            storage.write_json(resolved_path, notes)
            return note
    return None


def confirm_note(note_id: str, path: str | None = None) -> dict | None:
    """The bot found the note right again: it counts, and moves up in its group."""
    resolved_path = path or _default_path()
    notes = list_notes(resolved_path)
    for note in notes:
        if note["id"] == note_id:
            note["confirmations"] = note.get("confirmations", 0) + 1
            note["confirmed_at"] = _now()
            storage.write_json(resolved_path, notes)
            return note
    return None


def delete_note(note_id: str, path: str | None = None) -> bool:
    resolved_path = path or _default_path()
    notes = list_notes(resolved_path)
    remaining = [n for n in notes if n["id"] != note_id]
    if len(remaining) == len(notes):
        return False
    storage.write_json(resolved_path, remaining)
    return True


# --- topics ---------------------------------------------------------------------------------------

def topic_names(path: str | None = None) -> list[str]:
    """The topics the notes have, as they spell them."""
    seen: dict[str, str] = {}
    for note in list_notes(path):
        if note.get("scope") == "topic" and note.get("topic"):
            seen.setdefault(_fold(note["topic"]), note["topic"])
    return sorted(seen.values(), key=_fold)


def topic_keywords(path: str | None = None) -> dict[str, list[str]]:
    """Each topic's keywords besides its name ("Kim": ["Kim Pine"]), by name."""
    return storage.read_json(path or _topics_path(), default={})


def set_topic_keywords(name: str, keywords: list[str], path: str | None = None) -> list[str]:
    words = list(dict.fromkeys(" ".join(k.split()) for k in keywords if k and k.strip()))[:20]
    found = topic_keywords(path)
    if words:
        found[name] = words
    else:
        found.pop(name, None)
    storage.write_json(path or _topics_path(), found)
    return words


def clean_topics(names: list[str], path: str | None = None) -> list[str]:
    """A conversation's topics as the bot declared them, spelled as the notes spell them."""
    known = {_fold(n): n for n in topic_names(path)}
    out: dict[str, str] = {}
    for name in names:
        name = " ".join((name or "").split())[:MAX_TOPIC_LENGTH]
        if name:
            out.setdefault(_fold(name), known.get(_fold(name), name))
    return list(out.values())[:MAX_TOPICS]


def active_topics(declared: list[str], texts: list[str], path: str | None = None,
                  keywords_path: str | None = None) -> list[str]:
    """The topics that apply now: the conversation's own, then those whose name or a keyword
    appears in the texts (the latest messages, the title), as a whole word."""
    haystack = _fold("\n".join(t for t in texts if t))
    keywords = {_fold(k): v for k, v in topic_keywords(keywords_path).items()}
    active = {_fold(t): t for t in clean_topics(declared, path)}
    for name in topic_names(path):
        if _fold(name) in active:
            continue
        words = [name, *keywords.get(_fold(name), [])]
        if any(re.search(rf"(?<!\w){re.escape(_fold(w))}(?!\w)", haystack) for w in words if w.strip()):
            active[_fold(name)] = name
    return list(active.values())


# --- what a turn carries ---------------------------------------------------------------------------

def label(note: dict) -> str:
    """The note's group: "general", "family krea-2", "topic Kim"."""
    scope = note.get("scope") or "general"
    if scope == "family":
        return f"family {note.get('family')}"
    if scope == "topic":
        return f"topic {note.get('topic')}"
    return "general"


def notes_for_turn(notes: list[dict], covers, topics: list[str],
                   budget: int = NOTES_BUDGET) -> tuple[list[dict], dict[str, int]]:
    """The notes a turn carries, in groups -- general, the conversation's family (covers(folder)),
    its active topics in order -- each group's most recently confirmed first, within budget
    characters; and how many notes each group left out has ("topic Ramona": 12), the groups that
    don't apply now included."""
    def recent_first(group: list[dict]) -> list[dict]:
        return sorted(group, key=lambda n: n.get("confirmed_at") or n.get("created_at") or "", reverse=True)

    topic_order = {_fold(t): i for i, t in enumerate(topics)}
    general = recent_first([n for n in notes if (n.get("scope") or "general") == "general"])
    family = recent_first([n for n in notes if n.get("scope") == "family" and covers(n.get("family") or "")])
    topical = sorted((n for n in notes if n.get("scope") == "topic" and _fold(n.get("topic") or "") in topic_order),
                     key=lambda n: topic_order[_fold(n["topic"])])
    by_topic: dict[str, list[dict]] = {}
    for note in topical:
        by_topic.setdefault(_fold(note["topic"]), []).append(note)
    ordered = general + family + [n for group in by_topic.values() for n in recent_first(group)]
    shown, used = [], 0
    for note in ordered:
        if used + len(note["text"]) + 8 > budget:
            continue
        shown.append(note)
        used += len(note["text"]) + 8
    shown_ids = {n["id"] for n in shown}
    left: dict[str, int] = {}
    for note in notes:
        if note["id"] not in shown_ids:
            left[label(note)] = left.get(label(note), 0) + 1
    # Shown in group order and, within a group, oldest first: the numbers stay put when a note is
    # confirmed (Claude Code keeps the numbering it was sent until the notes change).
    groups = {"general": 0, "family": 1, "topic": 2}
    shown.sort(key=lambda n: (groups.get(n.get("scope") or "general", 0),
                              topic_order.get(_fold(n.get("topic") or ""), 0), n.get("created_at") or ""))
    return shown, left

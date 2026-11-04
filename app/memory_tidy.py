"""Ordenar la memoria: the bot reviews all its notes at once and proposes changes -- rewrite,
split, merge, forget, move to another scope -- each with its reason; the user picks which to
apply in Ajustes, and only those are.

`propose` asks the model (Claude Code, on the subscription, or one on OpenRouter) with every note
numbered, its scope, date and confirmations, the families' model folders and the topics, and the
rules a note must keep (one claim, short, standing on its own, in the right scope). Its answer --
JSON operations -- is checked (`_validate`): notes that exist, each used once, results that keep
the rules (chat.note_problem); what doesn't pass is dropped and counted. The plan waits in
memory_tidy.json, each operation with the notes before and after, until applied or discarded.
`apply` writes the accepted ones at once, keeping each note's date, conversation and
confirmations (a merge adds them up), skips any whose notes changed since, and keeps what was
there before in memory_tidy_undo.json: `undo` puts it back (with any note written since).
"""
import json
import logging
import re
import uuid
from datetime import datetime, timezone

from . import chat, claude_code, memory, storage
from .config import get_settings
from .openrouter_client import StreamEnd, stream_chat

logger = logging.getLogger(__name__)

OPERATIONS = ("rewrite", "split", "merge", "forget", "move")

SYSTEM = """You keep the memory of a prompt-crafting assistant that works with its user on image generation \
(ComfyUI): short notes it carries into its conversations. You review them all at once and propose the changes \
that make them correct, short and useful. You answer with JSON only."""

INSTRUCTIONS = """These are all the notes, numbered, each with its scope, the day it was written and how many \
times it proved right again since.

What a note must be:
- One claim, at most 200 characters, telegraphic. Several findings are several notes.
- Standing on its own in any conversation: never an image number ("#12", "the third one") or "this \
conversation"; say what was seen instead, and how many images back it ("2 of 2") when the note says so.
- In the scope where it applies: "general" (the user's tastes and way of working, true everywhere), "family" \
(what works or fails with a model family: "family" is its model folder, one of: {folders}) or "topic" (a \
character, a series, a project or a theme: "topic" is its name; the topics so far: {topics}).
- In the language it is written in.

Propose, as operations:
- "rewrite": a note that breaks those rules, or reads cut off, unclear or wrong: {{"op": "rewrite", "note": n, \
"text": "...", and optionally its new scope}}.
- "split": a note that holds several claims: {{"op": "split", "note": n, "into": [{{"text": "...", "scope": \
"...", "family" or "topic": "..."}}, ...]}}.
- "merge": notes that say the same, or the same finding at different times: {{"op": "merge", "notes": [n, m], \
"text": "...", and optionally its scope}}, with the evidence of both.
- "forget": a note that is superseded, contradicted by a later one, or not lasting (details of one image): \
{{"op": "forget", "note": n}}.
- "move": a note in the wrong scope: {{"op": "move", "note": n, "scope": "...", "family" or "topic": "..."}}.
Every operation has "why": the reason in a few words, in Spanish (the user reads it).

Rules: each note in one operation at most; leave out the notes that are fine. Don't invent facts or evidence: a \
note that reads cut off keeps what it says, finished only where the ending is evident. When two notes \
contradict each other, the later one wins unless the earlier has clearly more evidence; say so in "why". A \
scope is "general", "family" or "topic", and a family is one of the folders above.

Answer with one JSON object and nothing else: {{"operations": [...]}}.

The notes:
{notes}"""


def _plan_path() -> str:
    return f"{get_settings().data_dir}/memory_tidy.json"


def _undo_path() -> str:
    return f"{get_settings().data_dir}/memory_tidy_undo.json"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class TidyError(RuntimeError):
    """Something to show the user: the model failed, or answered nothing usable."""


def _listing(notes: list[dict]) -> str:
    lines = []
    for number, note in enumerate(notes, 1):
        day = (note.get("created_at") or "")[:10]
        confirmed = note.get("confirmations") or 0
        lines.append(f"[{number}] ({memory.label(note)}; {day}; confirmed {confirmed} times) {note['text']}")
    return "\n".join(lines)


def _ask(engine: str, model: str, prompt: str) -> tuple[str, float | None]:
    """The model's answer, and its cost (an estimate on Claude Code)."""
    context = {"mode": "memory_tidy"}
    if engine == claude_code.ENGINE:
        try:
            text, info = claude_code.ask(SYSTEM, prompt, model or "sonnet", context)
        except claude_code.AskError as e:
            raise TidyError(str(e))
        return text, info.get("estimated_cost")
    pieces, cost = [], None
    try:
        for item in stream_chat(get_settings().openrouter_api_key, model or get_settings().default_llm_model,
                                [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
                                temperature=0.2, context=context):
            if isinstance(item, StreamEnd):
                cost = item.cost
            else:
                pieces.append(item)
    except RuntimeError as e:
        raise TidyError(str(e))
    return "".join(pieces), cost


def _json(text: str) -> dict:
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise TidyError("The model didn't answer with the changes (no JSON). Try again.")
    try:
        found = json.loads(text[start:end + 1])
    except ValueError:
        raise TidyError("The model's answer wasn't valid JSON. Try again.")
    if not isinstance(found, dict) or not isinstance(found.get("operations"), list):
        raise TidyError("The model's answer had no list of operations. Try again.")
    return found


def _result(item: dict, default: dict | None, folders: list[str]) -> dict | None:
    """A note an operation leaves: {"text", "scope": {...}}, or None when it breaks the rules."""
    text = " ".join(str(item.get("text") or "").split())
    if not text or chat.note_problem(text):
        return None
    given = item.get("scope") or item.get("family") or item.get("topic")
    scope = memory.clean_scope(item.get("scope"), item.get("family"), item.get("topic")) if given else default
    if scope is None or (scope["scope"] == "family" and folders and scope["family"] not in folders):
        return None
    return {"text": text, "scope": scope}


def _scope_of(note: dict) -> dict:
    return memory.clean_scope(note.get("scope"), note.get("family"), note.get("topic"))


def _validate(raw: list, notes: list[dict], folders: list[str]) -> tuple[list[dict], int]:
    """The operations that hold, as {kind, ids, before, after, why}; and how many were dropped."""
    by_number = {n: note for n, note in enumerate(notes, 1)}
    used: set[str] = set()
    kept, dropped = [], 0

    def number(value):
        return value if isinstance(value, int) and value in by_number else None

    for op in raw:
        kind = op.get("op") if isinstance(op, dict) else None
        why = " ".join(str((op or {}).get("why") or "").split())[:300] if isinstance(op, dict) else ""
        targets, after = [], []
        if kind in ("rewrite", "split", "forget", "move"):
            n = number(op.get("note"))
            targets = [by_number[n]] if n else []
        elif kind == "merge":
            numbers = [number(v) for v in op.get("notes") or []]
            targets = [by_number[n] for n in dict.fromkeys(numbers)] if numbers and all(numbers) else []
            if len(targets) < 2:
                targets = []
        if not targets or any(t["id"] in used for t in targets):
            dropped += 1
            continue
        first = targets[0]
        if kind == "rewrite":
            result = _result(op, _scope_of(first), folders)
            after = [result] if result and (result["text"] != first["text"]
                                            or memory.label(result["scope"]) != memory.label(first)) else []
        elif kind == "split":
            parts = [_result(p, _scope_of(first), folders) for p in op.get("into") or [] if isinstance(p, dict)]
            after = parts if len(parts) >= 2 and all(parts) else []
        elif kind == "merge":
            result = _result(op, _scope_of(first), folders)
            after = [result] if result else []
        elif kind == "move":
            if op.get("scope") or op.get("family") or op.get("topic"):
                scope = memory.clean_scope(op.get("scope"), op.get("family"), op.get("topic"))
                ok = not (scope["scope"] == "family" and folders and scope["family"] not in folders)
                if ok and memory.label(scope) != memory.label(first):
                    after = [{"text": first["text"], "scope": scope}]
        if kind != "forget" and not after:
            dropped += 1
            continue
        used.update(t["id"] for t in targets)
        kept.append({
            "kind": kind, "ids": [t["id"] for t in targets],
            "before": [{"id": t["id"], "text": t["text"], "label": memory.label(t)} for t in targets],
            "after": [{"text": a["text"], "scope": a["scope"], "label": memory.label(a["scope"])} for a in after],
            "why": why,
        })
    return kept, dropped


def propose(engine: str, model: str = "") -> dict:
    """Ask for the review and keep the plan for the user to pick from."""
    notes = memory.list_notes()
    if not notes:
        raise TidyError("There are no notes to tidy.")
    from . import downloads  # the families' model folders, when ComfyUI's models folder is set
    folders = sorted({f for names in downloads.family_folders().values() for f in names}) or sorted(
        {n["family"] for n in notes if n.get("family")})
    prompt = INSTRUCTIONS.format(folders=", ".join(folders) or "(none known)",
                                 topics=", ".join(memory.topic_names()) or "(none yet)", notes=_listing(notes))
    text, cost = _ask(engine, model, prompt)
    operations, dropped = _validate(_json(text)["operations"], notes, folders)
    plan = {"id": uuid.uuid4().hex, "created_at": _now(), "engine": engine, "model": model, "cost": cost,
            "notes": len(notes), "operations": operations, "dropped": dropped}
    storage.write_json(_plan_path(), plan)
    logger.info("memory tidy proposed engine=%s model=%s operations=%d dropped=%d", engine, model,
                len(operations), dropped)
    return plan


def pending() -> dict | None:
    return storage.read_json(_plan_path(), default=None)


def discard() -> None:
    storage.remove(_plan_path())


def can_undo() -> bool:
    return storage.read_json(_undo_path(), default=None) is not None


def apply(accepted: list[int]) -> dict:
    """Apply the accepted operations (by their index in the plan), all at once."""
    plan = pending()
    if plan is None:
        raise TidyError("There is no proposal to apply.")
    notes = memory.list_notes()
    by_id = {n["id"]: n for n in notes}
    replaced: dict[str, list[dict]] = {}  # the first note an operation touches -> what takes its place
    removed: set[str] = set()
    produced: list[str] = []
    applied = skipped = 0
    for index in dict.fromkeys(accepted):
        if not isinstance(index, int) or not 0 <= index < len(plan["operations"]):
            continue
        op = plan["operations"][index]
        # A note edited, moved or deleted since the proposal: that operation no longer applies.
        if any(by_id.get(b["id"], {}).get("text") != b["text"] for b in op["before"]):
            skipped += 1
            continue
        targets = [by_id[i] for i in op["ids"]]
        first = targets[0]
        new = []
        for item in op["after"]:
            note = {**{k: first.get(k) for k in ("created_at", "conversation_id", "confirmed_at")},
                    "id": first["id"] if op["kind"] in ("rewrite", "move") else str(uuid.uuid4()),
                    "text": item["text"], "confirmations": first.get("confirmations", 0)}
            if op["kind"] == "merge":
                note["created_at"] = min(t.get("created_at") or "" for t in targets)
                note["confirmed_at"] = max(t.get("confirmed_at") or t.get("created_at") or "" for t in targets)
                note["confirmations"] = sum(t.get("confirmations", 0) for t in targets)
            for key in ("scope", "family", "topic"):
                note.pop(key, None)
            note.update(item["scope"])
            new.append(note)
            if note["id"] != first["id"]:
                produced.append(note["id"])
        replaced[first["id"]] = new
        removed.update(op["ids"][1:])
        applied += 1
    result = []
    for note in notes:
        if note["id"] in replaced:
            result += replaced[note["id"]]
        elif note["id"] not in removed:
            result.append(note)
    storage.write_json(_undo_path(), {"at": _now(), "previous": notes, "produced": produced})
    memory.replace_all(result)
    discard()
    logger.info("memory tidy applied operations=%d skipped=%d notes %d -> %d", applied, skipped, len(notes),
                len(result))
    return {"applied": applied, "skipped": skipped, "before": len(notes), "after": len(result)}


def undo() -> dict:
    """Put the notes back as they were before the last applied tidy (keeping any written since)."""
    saved = storage.read_json(_undo_path(), default=None)
    if saved is None:
        raise TidyError("There is nothing to undo.")
    previous_ids = {n["id"] for n in saved["previous"]}
    produced = set(saved.get("produced") or [])
    since = [n for n in memory.list_notes() if n["id"] not in previous_ids and n["id"] not in produced]
    memory.replace_all(saved["previous"] + since)
    storage.remove(_undo_path())
    return {"notes": len(saved["previous"]) + len(since)}

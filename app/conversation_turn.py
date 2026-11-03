"""One turn of the Conversar bot: from the conversation as stored to its reply, stored.

A turn streams the reply; when ComfyUI is available and the model takes tools, the bot
may call them (chat_tools.py), and each call is answered and the reply continues in a
new request -- up to MAX_STEPS requests per turn, the last one without tools. Each step
is a request of its own in the request log.

Turns of one conversation never overlap: a per-conversation lock is held from reading
the conversation to storing the reply. That also keeps a run's results (runs.py) from
landing before the reply that launched it. When a run finishes, its results message is
appended and, if nothing else is pending there, the bot gets an automatic turn to
comment on it, without tools that launch anything.

run_turn yields events: {"type": "delta", "text"} as the reply arrives, {"type": "tool",
"part"} for each tool call, and finally {"type": "done", "message", "conversation"} or
{"type": "error", "detail", "request_id"}.
"""
import base64
import json
import logging
import threading
import uuid
from contextlib import contextmanager
from pathlib import Path

from . import catalog, chat, chat_tools, comfy, conversations, families, history, memory, runs, workflows
from .characters import list_characters
from .config import get_settings
from .openrouter_client import StreamEnd, list_models, stream_chat

logger = logging.getLogger("app.conversations")

MAX_STEPS = 6
LOCK_WAIT_SECONDS = 600

_locks: dict[str, threading.Lock] = {}
_locks_guard = threading.Lock()
_automatic: set[str] = set()  # conversations whose bot is commenting on results right now


class Busy(RuntimeError):
    pass


@contextmanager
def turn_lock(conversation_id: str, timeout: float = LOCK_WAIT_SECONDS):
    with _locks_guard:
        lock = _locks.setdefault(conversation_id, threading.Lock())
    if not lock.acquire(timeout=timeout):
        raise Busy("The conversation is busy with another reply. Try again in a moment.")
    try:
        yield
    finally:
        lock.release()


def is_commenting(conversation_id: str) -> bool:
    return conversation_id in _automatic


def _image_loader(conversation_id: str):
    def load(image: dict) -> str | None:
        found = conversations.image_file(conversation_id, image["id"], small=True)
        if found is None:
            return None
        path, content_type = found
        return f"data:{content_type};base64,{base64.b64encode(Path(path).read_bytes()).decode('ascii')}"

    return load


def with_cache_breakpoints(model: str, messages: list[dict], history_length: int) -> list[dict]:
    """Mark the system prompt and the end of the stored conversation as cacheable, for
    Anthropic models (others cache on their own or not at all): every step of a turn, and
    the next turn within a few minutes, re-send them, and cached input costs a tenth."""
    if not model.startswith("anthropic/"):
        return messages
    marked = [dict(m) for m in messages]
    for index in {0, history_length - 1}:
        if not 0 <= index < len(marked) or marked[index]["role"] not in ("system", "user"):
            continue
        content = marked[index]["content"]
        parts = [{"type": "text", "text": content}] if isinstance(content, str) else [dict(p) for p in content]
        parts[-1] = {**parts[-1], "cache_control": {"type": "ephemeral"}}
        marked[index]["content"] = parts
    return marked


def _model_takes_tools(model: str) -> bool:
    try:
        known = {m["id"]: m for m in list_models()}
    except RuntimeError:
        return True  # can't tell: try, and the request log will show it if not
    entry = known.get(model)
    return entry is None or entry.get("supports_tools", True)


def toolbox_for(conversation: dict) -> chat_tools.ToolBox | None:
    client = comfy.client()
    if client is None or not _model_takes_tools(conversation["llm_model"]):
        return None
    return chat_tools.ToolBox(conversation["id"], client)


def _library_entry(workflow: dict) -> dict:
    from .comfy_graph import Graph
    folder = Graph(workflow["workflow"]).base_model_folder()
    family = families.family_for_folder(folder) if folder else None
    return {"name": workflow["name"], "description": workflow["description"],
            "family": family["name"] if family else None}


def _selected_characters(character_ids: list[str]) -> list[dict]:
    return [c for c in list_characters() if c["id"] in character_ids]


def _unanswered(conversation: dict) -> bool:
    return bool(conversation["messages"]) and conversation["messages"][-1]["role"] in ("user", "results")


def run_turn(conversation_id: str, automatic: bool = False):
    try:
        with turn_lock(conversation_id):
            if automatic:
                _automatic.add(conversation_id)
            try:
                yield from _turn(conversation_id, automatic)
            finally:
                _automatic.discard(conversation_id)
    except Busy as e:
        yield {"type": "error", "detail": str(e)}


def _turn(conversation_id: str, automatic: bool):
    conversation = conversations.get(conversation_id)
    if conversation is None:
        yield {"type": "error", "detail": "The conversation was deleted."}
        return
    last = conversation["messages"][-1] if conversation["messages"] else None
    if automatic and (last is None or last["role"] != "results"):
        return  # the user wrote since: that turn sees the results
    if not _unanswered(conversation):
        yield {"type": "error", "detail": "The last message already has a reply"}
        return
    family = families.get_family(conversation["family_id"])
    if family is None:
        yield {"type": "error", "detail": "The conversation's model family no longer exists."}
        return

    characters = _selected_characters(conversation["character_ids"])
    notes = memory.list_notes()
    toolbox = toolbox_for(conversation)
    library = [_library_entry(w) for w in workflows.list_workflows()]
    models_section = None
    if toolbox:
        catalog.refresh_in_background(toolbox.client)  # once a day, or the first time
        models_section = catalog.prompt_section()
    system = chat.build_system_prompt(
        family, characters, notes, needs_title=not conversation["title"],
        workflows=library if toolbox else None, models=models_section,
    )
    load_image = _image_loader(conversation_id)
    model, temperature = conversation["llm_model"], conversation["temperature"]
    api_key = get_settings().openrouter_api_key
    logger.info(
        "conversation reply conversation_id=%s family_id=%s model=%s messages=%d tools=%s automatic=%s",
        conversation_id, family["id"], model, len(conversation["messages"]), bool(toolbox), automatic,
    )

    steps: list[dict] = []  # {"text", "tool_calls", "results", "parts"}
    ends: list[StreamEnd] = []
    failure = None
    history_length = 1 + len(chat.build_llm_messages(conversation, load_image, tools=bool(toolbox)))
    for index in range(MAX_STEPS):
        messages = [
            {"role": "system", "content": system},
            *chat.build_llm_messages(
                conversation, load_image, tools=bool(toolbox),
                current_steps=[{"content": s["text"], "tool_calls": s["tool_calls"], "results": s["results"]}
                               for s in steps],
            ),
        ]
        messages = with_cache_breakpoints(model, messages, history_length)
        no_more_tools = automatic or index == MAX_STEPS - 1
        context = {"mode": "conversation", "conversation_id": conversation_id, "step": index + 1}
        if automatic:
            context["automatic"] = True
        chunks, end = [], StreamEnd(cost=None)
        try:
            for item in stream_chat(
                api_key, model, messages, temperature, context=context,
                tools=toolbox.definitions if toolbox else None,
                tool_choice="none" if toolbox and no_more_tools else None,
            ):
                if isinstance(item, StreamEnd):
                    end = item
                else:
                    chunks.append(item)
                    yield {"type": "delta", "text": item}
        except RuntimeError as e:
            logger.warning("conversation reply failed conversation_id=%s step=%d error=%s",
                           conversation_id, index + 1, e)
            failure = {"detail": str(e), "request_id": getattr(e, "request_id", None)}
            break
        ends.append(end)
        step = {"text": "".join(chunks), "tool_calls": [], "results": [], "parts": []}
        steps.append(step)
        if not (end.tool_calls and toolbox and not no_more_tools):
            break
        title = chat.TITLE_PATTERN.search("".join(s["text"] for s in steps))
        toolbox.title_hint = " ".join(title.group(1).split()) if title else ""
        for call in end.tool_calls:
            call = {"id": call.get("id") or f"call_{uuid.uuid4().hex[:12]}", "name": call["name"],
                    "arguments": call.get("arguments") or "{}"}
            result, part = toolbox.run(call["name"], call["arguments"])
            logger.info("conversation tool conversation_id=%s tool=%s error=%s",
                        conversation_id, call["name"], result.get("error"))
            step["tool_calls"].append(call)
            step["results"].append(json.dumps(result, ensure_ascii=False))
            step["parts"].append(part)
            yield {"type": "tool", "part": part}

    used_tools = any(s["tool_calls"] for s in steps)
    if failure and not used_tools:
        yield {"type": "error", **failure}
        return
    if not used_tools and not "".join(s["text"] for s in steps).strip():
        yield {
            "type": "error",
            "detail": "The model returned an empty reply. Try again.",
            "request_id": ends[-1].request_id if ends else None,
        }
        return
    # A run of another family's model may have switched the conversation's family meanwhile:
    # the reply's prompts are written (and kept) by the family it ended in.
    family = families.get_family((conversations.get(conversation_id) or {}).get("family_id", "")) or family
    updated = _store_reply(conversation_id, family, characters, notes, steps, ends, failure, automatic)
    if updated is None:
        yield {"type": "error", "detail": "The conversation was deleted while replying."}
        return
    yield {"type": "done", "message": updated["messages"][-1], "conversation": conversations.summary(updated)}


def _apply_note_changes(reply_updates, reply_forgets, notes_at_start: list[dict], conversation_id: str) -> list[dict]:
    """The bot's rewrites and deletions of existing notes, by the numbers it was shown."""
    changes = []
    by_number = {i: n for i, n in enumerate(notes_at_start, 1)}
    forgotten = set(reply_forgets)
    for number, text in reply_updates:
        note = by_number.get(number)
        if note is None or number in forgotten or text == note["text"]:
            continue
        if memory.update_note(note["id"], text) is not None:
            changes.append({"action": "update", "id": note["id"], "text": text, "previous": note["text"]})
    for number in reply_forgets:
        note = by_number.get(number)
        if note is not None and memory.delete_note(note["id"]):
            changes.append({"action": "forget", "id": note["id"], "text": note["text"]})
    if changes:
        logger.info("conversation notes changed conversation_id=%s changes=%d", conversation_id, len(changes))
    return changes


def _store_reply(conversation_id, family, characters, notes_at_start, steps, ends, failure, automatic):
    """Save the finished reply, its note changes, and (when it proposes prompts) a history entry."""
    conversation = conversations.get(conversation_id)
    if conversation is None:
        return None
    parts, new_notes, updates, forgets, title, contents = [], [], [], [], "", []
    stored_steps = []
    for step in steps:
        reply = chat.parse_reply(step["text"], family["has_negative_prompt"])
        title = title or reply.title
        parts += reply.parts + step["parts"]
        new_notes += reply.notes
        updates += reply.note_updates
        forgets += reply.note_forgets
        if reply.content:
            contents.append(reply.content)
        stored_steps.append({"content": reply.content, "tool_calls": step["tool_calls"], "results": step["results"]})

    created = [memory.create_note(note, conversation_id) for note in new_notes]
    note_changes = _apply_note_changes(updates, forgets, notes_at_start, conversation_id)
    costs = [e.cost for e in ends if e.cost is not None]
    cost = sum(costs) if costs else None
    request_ids = [e.request_id for e in ends if e.request_id]
    proposed = [{"positive_prompt": p["positive_prompt"], "negative_prompt": p["negative_prompt"]}
                for p in parts if p["type"] == "prompt"]

    history_id = None
    if proposed:
        last_user = next((m for m in reversed(conversation["messages"]) if m["role"] == "user"), None)
        entry = history.append_entry(
            mode="conversation",
            family_id=family["id"],
            family_name=family["name"],
            llm_model=conversation["llm_model"],
            vision_model=None,
            temperature=conversation["temperature"],
            user_input=(last_user or {}).get("content", ""),
            example_prompts="",
            previous_prompt=None,
            prompts=proposed,
            cost=cost,
            params={
                "conversation_id": conversation_id,
                "conversation_title": title or conversations.display_title(conversation),
                "request_ids": request_ids,
                **_character_params(characters),
            },
        )
        history_id = entry["id"]

    message = {
        "role": "assistant",
        "content": "\n\n".join(contents),
        "parts": parts,
        "notes": [{"id": n["id"], "text": n["text"]} for n in created],
        "cost": cost,
        "truncated": any(e.truncated for e in ends),
        "history_id": history_id,
        "request_id": request_ids[-1] if request_ids else (failure or {}).get("request_id"),
    }
    if len(request_ids) > 1:
        message["request_ids"] = request_ids
    if note_changes:
        message["note_changes"] = note_changes
    if any(s["tool_calls"] for s in stored_steps):
        message["steps"] = stored_steps
    if automatic:
        message["automatic"] = True
    if failure:
        message["error"] = failure["detail"]
    logger.info(
        "conversation reply stored conversation_id=%s prompts=%d notes=%d note_changes=%d steps=%d cost=%s",
        conversation_id, len(proposed), len(created), len(note_changes), len(steps), cost,
    )
    return conversations.append_message(conversation_id, message, title=title)


def _character_params(characters: list[dict]) -> dict:
    return {"character_names": [c["name"] for c in characters]} if characters else {}


def on_run_finished(conversation_id: str, run: dict) -> None:
    """A run is final: put its images in the chat, then let the bot comment on them --
    in one hold of the lock, and flagged as commenting before the images appear, so the
    chat never sees results with no comment coming (and stop checking back)."""
    # A retried download: its images go into the results message already in the chat.
    if conversations.set_results_images(conversation_id, run["n"], run["images"]):
        return
    will_comment = run["status"] != "cancelled" and not runs.tracker.pending_in(conversation_id)
    if will_comment:
        _automatic.add(conversation_id)
    try:
        # Long: a user's turn may be running, and the results must not be lost.
        with turn_lock(conversation_id, timeout=3 * LOCK_WAIT_SECONDS):
            stored = conversations.append_message(
                conversation_id, {"role": "results", "run": run["n"], "content": "", "images": run["images"]}
            )
            if stored is None or not will_comment:
                return
            for event in _turn(conversation_id, automatic=True):
                if event["type"] == "error":
                    logger.warning("automatic comment failed conversation_id=%s detail=%s",
                                   conversation_id, event["detail"])
    finally:
        _automatic.discard(conversation_id)


runs.tracker.on_finished = on_run_finished

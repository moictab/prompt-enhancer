"""One turn of the Conversar bot: from the conversation as stored to its reply, stored.

A turn streams the reply; when ComfyUI is available and the model takes tools, the bot
may call them (chat_tools.py), and each call is answered and the reply continues in a
new request -- up to MAX_STEPS requests per turn, the last one without tools -- except
after a step that just launched a run, which ends the turn. Each step is a request of
its own in the request log.

Turns of one conversation never overlap: a per-conversation lock is held from reading
the conversation to storing the reply. That also keeps a run's results (runs.py) from
landing before the reply that launched it. When a run finishes, its results message is
appended and, if nothing else is pending there and the conversation's automatic comments
are on, the bot gets an automatic turn to comment on it, without tools that launch
anything (unless it is working on its own).

run_turn yields events: {"type": "delta", "text"} as the reply arrives, {"type": "tool",
"part"} for each tool call, and finally {"type": "done", "message", "conversation"} or
{"type": "error", "detail", "request_id"}.

A turn the user asks for runs apart from the request that asked (start_turn): in a thread
of its own, publishing its events to a LiveTurn that any chat can follow from the start.
So the reply goes on, and is stored, when the chat that asked moves to another
conversation, reloads or closes -- several conversations can be replying at once -- and a
chat that opens the conversation meanwhile picks it up where it is. The bot's automatic
comment on results is a LiveTurn too.
"""
import base64
import json
import logging
import threading
import uuid
from contextlib import contextmanager
from pathlib import Path

from . import (catalog, chat, chat_tools, claude_code, comfy, conversations, families, history, memory,
               retry_later, runs, workflows)
from .config import get_settings
from .openrouter_client import StreamEnd, list_models, stream_chat

logger = logging.getLogger("app.conversations")

MAX_STEPS = 6
# A step that launches a run, and calls nothing else whose result the bot needs to read, ends
# the turn: asking the model to go on would cost a request for it to say "launched" (the
# bot says what it runs before the call; COMFY_MECHANICS tells it so).
LAUNCH_TOOLS = ("run_workflow", "calibrate_lora")
QUIET_TOOLS = (*LAUNCH_TOOLS, "keep_going", "save_workflow", "cancel_run")
LOCK_WAIT_SECONDS = 600
# A follower with nothing new for this long gets None (the route sends a blank line), so a
# connection waiting on a slow model isn't taken for a dead one.
KEEPALIVE_SECONDS = 15

_locks: dict[str, threading.Lock] = {}
_locks_guard = threading.Lock()
_automatic: set[str] = set()  # conversations whose bot is commenting on results right now


class Busy(RuntimeError):
    pass


class LiveTurn:
    """A turn being written: its events so far, for whoever follows it."""

    def __init__(self, conversation_id: str, automatic: bool = False):
        self.conversation_id = conversation_id
        self.automatic = automatic
        self.events: list[dict] = []
        self.finished = False
        self._condition = threading.Condition()

    def publish(self, event: dict) -> None:
        with self._condition:
            self.events.append(event)
            self._condition.notify_all()

    def finish(self) -> None:
        with self._condition:
            self.finished = True
            self._condition.notify_all()

    def follow(self, keepalive: float = KEEPALIVE_SECONDS):
        """Every event from the first, then each as it comes, until the turn ends; None
        after `keepalive` seconds with nothing new."""
        index = 0
        while True:
            with self._condition:
                if index == len(self.events) and not self.finished:
                    self._condition.wait(timeout=keepalive)
                pending = self.events[index:]
                finished = self.finished
            index += len(pending)
            if pending:
                yield from pending
            elif finished:
                return
            else:
                yield None


_live: dict[str, LiveTurn] = {}
_live_guard = threading.Lock()


def live_turn(conversation_id: str) -> LiveTurn | None:
    return _live.get(conversation_id)


def is_replying(conversation_id: str) -> bool:
    """The bot is answering the user there right now (not commenting on results)."""
    live = _live.get(conversation_id)
    return live is not None and not live.automatic


def is_busy(conversation_id: str) -> bool:
    """A turn is under way there, or the bot is about to comment on results: a new message
    would land before its reply."""
    return conversation_id in _live or conversation_id in _automatic


def _register(live: LiveTurn) -> None:
    with _live_guard:
        _live[live.conversation_id] = live


def _unregister(live: LiveTurn) -> None:
    with _live_guard:
        if _live.get(live.conversation_id) is live:
            del _live[live.conversation_id]
        if live.automatic:
            _automatic.discard(live.conversation_id)


def _publish_all(live: LiveTurn, events) -> None:
    """Publish a turn's events as they come. The turn stops being live just before its last
    event goes out, so whoever reads that one finds the conversation free."""
    try:
        for event in events:
            if event["type"] in ("done", "error"):
                _unregister(live)
            live.publish(event)
    except Exception:
        logger.exception("conversation turn failed conversation_id=%s", live.conversation_id)
        _unregister(live)
        live.publish({"type": "error", "detail": "The reply failed unexpectedly. Try again."})
    finally:
        _unregister(live)
        live.finish()


def start_turn(conversation_id: str) -> LiveTurn:
    """Answer the conversation's last message in a thread of its own; returns the turn to
    follow. Raises Busy if a turn is already under way there."""
    with _live_guard:
        if conversation_id in _live or conversation_id in _automatic:
            raise Busy("The bot is still replying in this conversation. Wait for it to finish.")
        live = LiveTurn(conversation_id)
        _live[conversation_id] = live
    threading.Thread(
        target=_publish_all, args=(live, run_turn(conversation_id)),
        name=f"turn-{conversation_id[:8]}", daemon=True,
    ).start()
    return live


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


def with_cache_breakpoints(model: str, messages: list[dict], history_length: int,
                           volatile_tail: int = 0) -> list[dict]:
    """Mark where Anthropic models may cache the request (others cache on their own or not
    at all). Cached input costs a tenth; writing it, a quarter more. A cache entry serves a
    later request whose start is exactly the same up to that mark, so the marks go where
    the next request will match:
    - the system prompt (with the tools before it), the same every turn;
    - the latest user-side message (the stored conversation's end, history_length - 1),
      before its volatile tail (the #N images and the context, which change every turn):
      next turn, this message has no tail, and everything up to here is read from cache;
    - its volatile tail, which the turn's later steps re-send;
    - the end of the user-side message before it: what the previous turn cached.
    Four marks at most, Anthropic's limit."""
    if not model.startswith("anthropic/"):
        return messages
    marked = [dict(m) for m in messages]
    latest = history_length - 1
    marks = {0: [-1]}
    if 0 < latest < len(marked) and marked[latest]["role"] == "user":
        marks[latest] = [-1 - volatile_tail, -1] if volatile_tail else [-1]
        previous = next((i for i in range(latest - 1, 0, -1) if marked[i]["role"] == "user"), None)
        if previous is not None:
            marks[previous] = [-1]
    for index, positions in marks.items():
        content = marked[index]["content"]
        parts = [{"type": "text", "text": content}] if isinstance(content, str) else [dict(p) for p in content]
        for position in positions:
            if -len(parts) <= position:
                parts[position] = {**parts[position], "cache_control": {"type": "ephemeral"}}
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
    if client is None:
        return None
    # Claude Code always calls tools; a model on OpenRouter only if the model list says it can.
    if not claude_code.engine(conversation) and not _model_takes_tools(conversation["llm_model"]):
        return None
    return chat_tools.ToolBox(conversation["id"], client)


def _library_entry(workflow: dict) -> dict:
    from .comfy_graph import Graph
    graph = Graph(workflow["workflow"])
    folder = graph.base_model_folder()
    family = families.family_for_folder(folder) if folder else None
    return {"name": workflow["name"], "description": workflow["description"],
            "family": family["name"] if family else None, "size": graph.shape(),
            "model": ", ".join(graph.base_models()) or None}


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

    notes = memory.list_notes()
    toolbox = toolbox_for(conversation)
    if toolbox:
        toolbox.automatic = automatic
    library = [_library_entry(w) for w in workflows.list_workflows()]
    models_section = installed = None
    if toolbox:
        catalog.refresh_in_background(toolbox.client)  # metadata: once a day, or the first time
        catalog.ensure_fresh(toolbox.client)  # file names: before every turn, if a couple of minutes old
        models_section = catalog.prompt_section()
        installed = catalog.load()["entries"] or None
    family_names = [f["name"] for f in families.list_families()]
    # Working on its own: an automatic turn (its look at a run's images) may launch the next run,
    # while the budget lasts.
    working = conversations.autonomy(conversation)
    autonomous = automatic and working is not None and not _budget_spent(working)
    # Taking stock (balance.py) is a look back: nothing to call, let alone launch.
    taking_stock = bool(last.get("balance"))
    # What changes every turn goes at the end of the latest message, after the cached part.
    context_parts = {
        "notes": notes, "models": models_section, "needs_title": not conversation["title"],
        "autonomy": chat.autonomy_text(working, automatic, _budget_spent(working)) if working else None,
        "comments_off": bool(toolbox) and not conversations.comments_on_results(conversation),
        # Its look at a run's results launches nothing: it shouldn't say it will.
        "looking": bool(toolbox) and automatic and not autonomous,
        # The user discarded its reply to this message ("Rehacer") and wants another.
        "redo": bool(last.get("redo")),
    }
    load_image = _image_loader(conversation_id)
    on_claude = claude_code.engine(conversation)
    logger.info(
        "conversation reply conversation_id=%s family_id=%s model=%s messages=%d tools=%s automatic=%s",
        conversation_id, family["id"],
        f"claude-code/{claude_code.model_for(conversation)}" if on_claude else conversation["llm_model"],
        len(conversation["messages"]), bool(toolbox), automatic,
    )

    extra = None
    if on_claude:
        steps, ends, failure, extra = yield from claude_code.run(
            conversation, family, toolbox, library, family_names, context_parts, load_image, installed,
            automatic=automatic, tools_allowed=not ((automatic and not autonomous) or taking_stock))
    else:
        system = chat.build_system_prompt(family, workflows=library if toolbox else None,
                                          families=family_names if toolbox else None)
        steps, ends, failure = yield from _openrouter_steps(
            conversation, system, chat.turn_context(**context_parts), toolbox, load_image, installed,
            automatic=automatic, no_tools=(automatic and not autonomous) or taking_stock)

    used_tools = any(s["tool_calls"] for s in steps)
    if failure and not used_tools:
        _settle_autonomy(conversation_id)
        if failure.get("retry_at"):  # Claude Code's limit: answered again once it resets
            conversations.set_retry(conversation_id, failure["retry_at"])
            retry_later.schedule(conversation_id, failure["retry_at"])
        yield {"type": "error", **failure}
        return
    if not used_tools and not "".join(s["text"] for s in steps).strip():
        _settle_autonomy(conversation_id)
        yield {
            "type": "error",
            "detail": "The model returned an empty reply. Try again.",
            "request_id": ends[-1].request_id if ends else None,
        }
        return
    # A run of another family's model may have switched the conversation's family meanwhile:
    # the reply's prompts are written (and kept) by the family it ended in.
    family = families.get_family((conversations.get(conversation_id) or {}).get("family_id", "")) or family
    updated = _store_reply(conversation_id, family, notes, steps, ends, failure, automatic, extra)
    if updated is None:
        yield {"type": "error", "detail": "The conversation was deleted while replying."}
        return
    if _settle_autonomy(conversation_id):
        updated = conversations.get(conversation_id) or updated
    yield {"type": "done", "message": updated["messages"][-1], "conversation": conversations.summary(updated)}


def _openrouter_steps(conversation: dict, system: str, turn_context: str | None, toolbox, load_image, installed,
                      *, automatic: bool, no_tools: bool):
    """The reply from the LLM on OpenRouter, in steps: yields its events, returns (steps, ends, failure)."""
    conversation_id = conversation["id"]
    model, temperature = conversation["llm_model"], conversation["temperature"]
    api_key = get_settings().openrouter_api_key
    steps: list[dict] = []  # {"text", "tool_calls", "results", "parts"}
    ends: list[StreamEnd] = []
    failure = None
    history = chat.build_llm_messages(conversation, load_image, tools=bool(toolbox), installed=installed,
                                      context=turn_context)
    history_length = 1 + len(history)
    for index in range(MAX_STEPS):
        messages = [
            {"role": "system", "content": system},
            *chat.build_llm_messages(
                conversation, load_image, tools=bool(toolbox),
                current_steps=[{"content": s["text"], "tool_calls": s["tool_calls"], "results": s["results"]}
                               for s in steps],
                installed=installed, context=turn_context,
            ),
        ]
        messages = with_cache_breakpoints(model, messages, history_length, history.volatile_tail)
        no_more_tools = no_tools or index == MAX_STEPS - 1
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
        toolbox.reply_text = "".join(s["text"] for s in steps)
        title = chat.TITLE_PATTERN.search(toolbox.reply_text)
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
        if _launched(step):
            break  # the run is under way: nothing left to say until its images arrive
    return steps, ends, failure


def _launched(step: dict) -> bool:
    """The step launched a run, every call in it went through (a failed one, or a family
    switch, comes back as an error for the bot to fix), and none of them returned anything
    the bot has to read."""
    names = [call["name"] for call in step["tool_calls"]]
    return (any(name in LAUNCH_TOOLS for name in names)
            and all(name in QUIET_TOOLS for name in names)
            and not any("error" in json.loads(result) for result in step["results"]))


def _budget_spent(working: dict) -> bool:
    """No images left, or not enough money left to look at another run's images."""
    return (working["images"] >= working["max_images"]
            or working["cost"] + working.get("last_turn_cost", 0) > working["max_cost"])


def _settle_autonomy(conversation_id: str) -> bool:
    """Working on its own ends with a turn that leaves nothing under way: the bot stopped
    launching (the goal is met, or it is out of ideas) or the budget ran out. Whether it ended."""
    working = conversations.autonomy(conversations.get(conversation_id) or {})
    if working is None or runs.tracker.pending_in(conversation_id):
        return False
    if working["images"] >= working["max_images"]:
        reason = "images"
    elif working["cost"] + working.get("last_turn_cost", 0) > working["max_cost"]:
        reason = "cost"
    else:
        reason = "done"
    conversations.end_autonomy(conversation_id, reason)
    logger.info("autonomy ended conversation_id=%s reason=%s images=%s cost=%s",
                conversation_id, reason, working["images"], working["cost"])
    return True


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


def _store_reply(conversation_id, family, notes_at_start, steps, ends, failure, automatic, extra=None):
    """Save the finished reply, its note changes, and (when it proposes prompts) a history entry.
    extra: fields for the message (Claude Code's: which engine wrote it, its estimated cost)."""
    conversation = conversations.get(conversation_id)
    if conversation is None:
        return None
    parts, new_notes, updates, forgets, title, contents, reviews = [], [], [], [], "", [], []
    stored_steps = []
    for step in steps:
        reply = chat.parse_reply(step["text"], family["has_negative_prompt"])
        title = title or reply.title
        parts += reply.parts + step["parts"]
        new_notes += reply.notes
        updates += reply.note_updates
        forgets += reply.note_forgets
        reviews += reply.reviews
        if reply.content:
            contents.append(reply.content)
        stored_steps.append({"content": reply.content, "tool_calls": step["tool_calls"], "results": step["results"]})

    created = [memory.create_note(note, conversation_id) for note in new_notes]
    _store_reviews(conversation_id, conversation, reviews)
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
    if extra:
        message.update(extra)
    logger.info(
        "conversation reply stored conversation_id=%s prompts=%d notes=%d note_changes=%d steps=%d cost=%s",
        conversation_id, len(proposed), len(created), len(note_changes), len(steps), cost,
    )
    stored = conversations.append_message(conversation_id, message, title=title)
    if stored is not None and conversations.autonomy(stored):
        conversations.add_autonomy_usage(conversation_id, turn_cost=cost or 0.0)
        stored = conversations.get(conversation_id) or stored
    return stored


def _store_reviews(conversation_id: str, conversation: dict, reviews: list[tuple[int, str, str]]) -> None:
    """The bot's verdict on each image, by the numbers it sees, kept with the image (and so
    with its run's recipe, for the catalog's per-file stats)."""
    if not reviews:
        return
    by_number = {n: image_id for image_id, n in chat.image_numbers(conversation["messages"]).items()}
    stored = 0
    for number, verdict, note in reviews:
        if number in by_number and conversations.set_review(conversation_id, by_number[number], verdict, note):
            stored += 1
    if stored:
        catalog.invalidate_stats()
        logger.info("conversation reviews stored conversation_id=%s reviews=%d", conversation_id, stored)


def on_run_finished(conversation_id: str, run: dict) -> None:
    """A run is final: put its images in the chat, then let the bot comment on them --
    in one hold of the lock, and flagged as commenting before the images appear, so the
    chat never sees results with no comment coming (and stop checking back). With the
    conversation's automatic comments off, the images wait for the user's next message."""
    # A run of a reply the user discarded ("Rehacer"): its images don't come into the chat.
    if run.get("discarded"):
        return
    # A retried download: its images go into the results message already in the chat.
    if conversations.set_results_images(conversation_id, run["n"], run["images"]):
        return
    will_comment = (run["status"] != "cancelled" and not runs.tracker.pending_in(conversation_id)
                    and conversations.comments_on_results(conversations.get(conversation_id) or {}))
    live = LiveTurn(conversation_id, automatic=True) if will_comment else None
    if will_comment:
        _automatic.add(conversation_id)
    try:
        # Long: a user's turn may be running, and the results must not be lost.
        with turn_lock(conversation_id, timeout=3 * LOCK_WAIT_SECONDS):
            if live:
                _register(live)  # before the images appear: a chat seeing them can follow the comment
            stored = conversations.append_message(
                conversation_id, {"role": "results", "run": run["n"], "content": "", "images": run["images"]}
            )
            if stored is None or not will_comment:
                return
            _publish_all(live, _logged(_turn(conversation_id, automatic=True), conversation_id))
    finally:
        _automatic.discard(conversation_id)
        if live:
            _unregister(live)
            live.finish()


def _logged(events, conversation_id: str):
    for event in events:
        if event["type"] == "error":
            logger.warning("automatic comment failed conversation_id=%s detail=%s",
                           conversation_id, event["detail"])
        yield event


runs.tracker.on_finished = on_run_finished

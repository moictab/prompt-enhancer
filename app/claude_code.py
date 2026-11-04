"""Conversar's bot played by Claude Code, under a Claude subscription, instead of an LLM on OpenRouter.

A conversation whose engine is "claude-code" is answered by running the Claude Code CLI
headless, once per turn: `claude -p` with stream-json in and out, resuming that conversation's
own session (one per conversation; the app picks its id), in an empty folder of its own, with:
- its own configuration folder (CLAUDE_CONFIG_DIR = CLAUDE_CODE_CONFIG_DIR): it uses the login
  made there -- an account of its own -- and none of the settings, plugins, hooks, memory or
  sessions of the Claude Code used elsewhere on the PC. ANTHROPIC_* and CLAUDE_CODE_* variables
  are kept out of its environment, so that login is what it uses.
- no tools but the bot's, over MCP at this app (bot_seat.py, a URL for this turn only):
  --tools "" removes the built-in ones (shell, files, web), --strict-mcp-config with
  --mcp-config leaves that one server, --allowedTools lets its tools run, --permission-mode
  dontAsk refuses the rest, --setting-sources "" reads no settings files. A session that starts
  with any other tool is stopped (_check_tools).
- the bot's system prompt: chat.system_sections, the same personality, family rules,
  library and mechanics as on OpenRouter, plus CLAUDE_CODE_MECHANICS. Claude Code records it
  when the session starts and keeps it, so what changes later -- the family, the library --
  comes in the turn's message (a section whose digest differs from
  the session's start). So do the notes and the installed models, sent again only when they
  changed, or every RESEND_EVERY turns: the session keeps what was sent before.

A turn sends the latest user-side message as chat.build_llm_messages makes it (a run's
results, the user's text, the images attached, the reactions) plus that context. A session's
first turn in a conversation that already has messages (it began on OpenRouter, or the
session was lost) first gets the conversation so far. The text streams into the chat as it is
written, each tool call shows as its part, and the reply is stored like the OpenRouter bot's
(conversation_turn._store_reply): blocks, notes, reviews, title, history. Its cost is the
CLI's estimate -- the subscription pays, not per request -- kept on the message apart from the
conversation's cost. Each turn is one entry in the request log. At most MAX_PARALLEL run at once.
"""
import hashlib
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

from . import bot_seat, chat, conversations, request_log, storage
from .config import get_settings
from .openrouter_client import StreamEnd

logger = logging.getLogger("app.conversations")

ENGINE = "claude-code"
MODELS = ("opus", "sonnet", "haiku")
SERVER_KEY = "pe"
TOOL_PREFIX = f"mcp__{SERVER_KEY}__"
MAX_TURNS = 40  # agentic steps in one reply
MAX_PARALLEL = 2  # Claude Code processes at once; more wait their turn
RESEND_EVERY = 8  # turns after which the notes and models are sent again even if unchanged
TURN_TIMEOUT_SECONDS = 1800
# Sections of the system prompt that may change during a conversation, and are then sent again.
CHANGING_SECTIONS = ("family", "library", "families")
# Claude Code ages sessions out after 30 days by default: these are conversations, kept longer.
SESSION_SETTINGS = {"cleanupPeriodDays": 3650}

CLAUDE_CODE_MECHANICS = """- Your tools reach you through the app: you have no others (no shell, files or web). Their names here start with mcp__pe__; these instructions call them by their short names.
- The app numbers your <prompt> blocks across the conversation, in order (1, 2, 3...): the app's context in the latest message says which numbers your previous reply's blocks got, so you can tell which one the user means by its number.
- New images come attached to their message; get_image shows any image of the conversation again by its number. get_balance_data lists every image the runs made here, with its recipe, your review and the user's reaction.
- Your notes and the installed models come in the app's context when they change: otherwise the latest list you got still holds, numbers included.
- Unlike what's said above, a launch here doesn't end your reply: launch every run you announced (one call each), then end it in one short sentence at most, since their images come in a later message."""

HISTORY_INTRO = ("[This conversation began before this session. Here it is so far, as the app keeps it: the "
                 "user's messages, your earlier replies (their prompt blocks numbered), the runs' results and "
                 "the images still in the window attached; any other image by number only (get_image shows it).]")
HISTORY_END = "[The conversation so far ends here. The latest message follows.]"
CHANGED_HEADING = ("## Changed Since This Session Started\nThese replace what your instructions say about "
                   "them: the conversation follows them now.")
UNCHANGED_NOTE = ("- Your notes and the installed models haven't changed since they last came: the latest list "
                  "you got still holds.")
NOT_LOGGED_IN = ("Claude Code isn't signed in for prompt-enhancer. In PowerShell: "
                 "$env:CLAUDE_CONFIG_DIR = \"{config}\"; claude auth login")
UNEXPECTED_TOOLS = "Claude Code started with tools it mustn't have ({tools}): stopped. Nothing ran."

_slots = threading.BoundedSemaphore(MAX_PARALLEL)


def engine(conversation: dict) -> bool:
    return conversation.get("engine") == ENGINE


def available() -> bool:
    return shutil.which(get_settings().claude_code_command) is not None


def model_for(conversation: dict) -> str:
    return conversation.get("claude_model") or get_settings().claude_code_model


def _digest(text: str | None) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()[:16]


def _folder(*parts: str) -> Path:
    # Absolute: the CLI runs in the workspace folder, where a path relative to the app's (DATA_DIR
    # is "data" by default) would point elsewhere.
    folder = Path(get_settings().data_dir, "claude-code", *parts).resolve()
    folder.mkdir(parents=True, exist_ok=True)
    return folder


# --- what Claude Code is sent ----------------------------------------------------------------------

def _blocks(content) -> list[dict]:
    """A chat message's content (build_llm_messages: a string, or text and image_url parts) as
    Claude's content blocks."""
    if isinstance(content, str):
        return [{"type": "text", "text": content}] if content else []
    out = []
    for part in content:
        if part["type"] == "text" and part.get("text"):
            out.append({"type": "text", "text": part["text"]})
        elif part["type"] == "image_url":
            header, _, data = part["image_url"]["url"].partition(",")
            media_type = header[len("data:"):].split(";")[0]
            out.append({"type": "image", "source": {"type": "base64", "media_type": media_type, "data": data}})
    return out


def _history(messages: list[dict]) -> list[dict]:
    """The conversation before the latest message, for a session that didn't see it."""
    blocks = [{"type": "text", "text": HISTORY_INTRO}]
    for message in messages:
        label = "[The user]" if message["role"] == "user" else "[You]"
        content = _blocks(message.get("content") or "")
        if content and content[0]["type"] == "text":
            content[0] = {"type": "text", "text": f"{label}\n{content[0]['text']}"}
        else:
            content.insert(0, {"type": "text", "text": label})
        blocks += content
    blocks.append({"type": "text", "text": HISTORY_END})
    return blocks


def _previous_numbers(conversation: dict) -> list[int]:
    """The numbers the app gave the prompt blocks of the bot's latest reply."""
    count, latest = 0, []
    for message in conversation["messages"]:
        if message["role"] != "assistant":
            continue
        prompts = sum(1 for p in message.get("parts") or [] if p["type"] == "prompt")
        latest = list(range(count + 1, count + prompts + 1))
        count += prompts
    return latest


def _context(conversation: dict, state: dict, sections: dict[str, str], parts: dict, new_session: bool):
    """The app's context for this turn (chat.turn_context, plus what changed since the session
    started and the numbering of the latest proposals); and what the state records of it."""
    notes_text = "\n".join(n["text"] for n in parts["notes"])
    digests = {"notes": _digest(notes_text), "models": _digest(parts.get("models"))}
    sent_before = state.get("context") or {}
    resend = (new_session or digests != sent_before
              or state.get("turns_since_context", 0) + 1 >= RESEND_EVERY)
    text = chat.turn_context(parts["notes"] if resend else [], parts.get("models") if resend else None,
                             needs_title=parts["needs_title"], autonomy=parts.get("autonomy"),
                             comments_off=parts.get("comments_off", False), looking=parts.get("looking", False),
                             redo=parts.get("redo", False))
    extra = []
    if not new_session:
        started = state.get("sections") or {}
        changed = [sections.get(name) or f"## {name.title()}\n(none now)" for name in CHANGING_SECTIONS
                   if _digest(sections.get(name)) != started.get(name, _digest(None))]
        if changed:
            extra.append("\n\n".join([CHANGED_HEADING, *changed]))
    if not resend:
        extra.append(UNCHANGED_NOTE)
    numbers = _previous_numbers(conversation)
    if numbers:
        listed = ", ".join(str(n) for n in numbers)
        extra.append(f"- Your previous reply's prompt blocks are numbered {listed}.")
    if extra:
        text = "\n\n".join([text or chat.CONTEXT_HEADING, *extra])
    recorded = {"context": digests if resend else sent_before,
                "turns_since_context": 0 if resend else state.get("turns_since_context", 0) + 1}
    return text, recorded


def build_message(conversation: dict, state: dict, sections: dict[str, str], parts: dict, load_image,
                  installed, new_session: bool) -> tuple[list[dict], dict]:
    """The user-side message of this turn, as Claude content blocks; and the context it records."""
    context, recorded = _context(conversation, state, sections, parts, new_session)
    replay = chat.build_llm_messages(conversation, load_image, tools=False, installed=installed, context=context)
    blocks = _blocks(replay[-1]["content"]) if replay and replay[-1]["role"] == "user" else []
    if new_session and len(replay) > 1:
        blocks = _history(replay[:-1]) + blocks
    return blocks, recorded


def args_for(model: str, session_id: str, resume: bool, files: dict[str, Path]) -> list[str]:
    command = shutil.which(get_settings().claude_code_command) or get_settings().claude_code_command
    return [
        command, "-p", "--input-format", "stream-json", "--output-format", "stream-json", "--verbose",
        "--include-partial-messages", "--model", model,
        "--tools", "", "--strict-mcp-config", "--mcp-config", str(files["mcp"]),
        "--allowedTools", f"{TOOL_PREFIX}*", "--permission-mode", "dontAsk", "--permission-prompts", "none",
        "--setting-sources", "", "--settings", str(files["settings"]),
        "--system-prompt-file", str(files["system"]), "--max-turns", str(MAX_TURNS),
        *(["--resume", session_id] if resume else ["--session-id", session_id]),
    ]


def environment() -> dict[str, str]:
    """The parent's environment without what would point Claude Code at another account or
    session, plus its own configuration folder."""
    env = {k: v for k, v in os.environ.items()
           if not (k.startswith("ANTHROPIC_") or k.startswith("CLAUDE_CODE_") or k in ("CLAUDECODE", "CLAUDE_CONFIG_DIR"))}
    env["CLAUDE_CONFIG_DIR"] = get_settings().claude_code_config_dir
    return env


def _spawn(args: list[str], env: dict[str, str], cwd: str) -> subprocess.Popen:
    flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
    return subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            cwd=cwd, env=env, text=True, encoding="utf-8", errors="replace", bufsize=1,
                            creationflags=flags)


def _write_files(folder: Path, system: str, seat: bot_seat.Seat) -> dict[str, Path]:
    files = {"system": folder / "system.md", "mcp": folder / "mcp.json", "settings": folder / "settings.json"}
    files["system"].write_text(system, encoding="utf-8")
    files["mcp"].write_text(json.dumps({"mcpServers": {SERVER_KEY: {
        "type": "http", "url": bot_seat.url(seat, get_settings().self_url)}}}), encoding="utf-8")
    files["settings"].write_text(json.dumps(SESSION_SETTINGS), encoding="utf-8")
    return files


def _logged_blocks(blocks: list[dict]) -> list[dict]:
    """The message for the request log: images as data URIs, which the log keeps once each."""
    out = []
    for block in blocks:
        if block["type"] == "image":
            source = block["source"]
            out.append({"type": "image", "url": f"data:{source['media_type']};base64,{source['data']}"})
        else:
            out.append(block)
    return out


def _check_tools(event: dict) -> str | None:
    unexpected = [t for t in event.get("tools") or [] if not t.startswith(TOOL_PREFIX)]
    return UNEXPECTED_TOOLS.format(tools=", ".join(unexpected)) if unexpected else None


def _failure_text(result: dict | None, stderr: str, code: int | None) -> str:
    text = " ".join(str((result or {}).get("result") or "").split())
    errors = (result or {}).get("errors")
    if not text and errors:
        text = " ".join(str(e) for e in errors)
    lowered = f"{text} {stderr}".lower()
    if "/login" in lowered or "not logged in" in lowered or "invalid api key" in lowered or "please run" in lowered:
        return NOT_LOGGED_IN.format(config=get_settings().claude_code_config_dir)
    if text:
        return f"Claude Code: {text}"[:600]
    tail = " ".join(stderr.split())[-400:]
    return f"Claude Code stopped (exit code {code}){': ' + tail if tail else ''}."


def limits_path() -> Path:
    return Path(get_settings().data_dir, "claude-code", "limits.json")


def save_limits(info: dict) -> None:
    """The subscription's usage, as the latest turn reported it (the account's, whichever
    conversation it came from): each window's share used and when it resets."""
    windows = {name: {"used": w.get("utilization"), "resets_at": w.get("resetsAt"), "status": w.get("status")}
               for name, w in (info.get("unifiedWindows") or {}).items() if isinstance(w, dict)}
    if not windows and info.get("rateLimitType"):
        windows[info["rateLimitType"]] = {"used": info.get("utilization"), "resets_at": info.get("resetsAt"),
                                          "status": info.get("status")}
    try:
        storage.write_json(str(limits_path()), {
            "checked_at": datetime.now(timezone.utc).isoformat(), "status": info.get("status"),
            "resets_at": info.get("resetsAt"), "windows": windows})
    except OSError as e:
        logger.warning("claude code limits not saved: %r", e)


def limits() -> dict | None:
    return storage.read_json(str(limits_path()), default=None)


LIMIT_WORDS = re.compile(r"usage limit|limit reached|hit your limit|out of usage", re.IGNORECASE)


def limit_resets(result: dict | None, rate_limit: dict | None, stderr: str) -> int | None:
    """When a turn failed because the subscription's limit was reached: when it resets (epoch
    seconds), if known."""
    info = rate_limit or {}
    windows = [w for w in (info.get("unifiedWindows") or {}).values() if isinstance(w, dict)]
    if info.get("status") == "rejected":
        blocked = [w.get("resetsAt") for w in windows if w.get("status") == "rejected" and w.get("resetsAt")]
        resets = max(blocked) if blocked else info.get("resetsAt")
        return int(resets) if resets else None
    text = f"{(result or {}).get('result') or ''} {stderr}"
    if not LIMIT_WORDS.search(text):
        return None
    stamp = re.search(r"\|(\d{10})\b", text)  # "Claude AI usage limit reached|1793809800"
    if stamp:
        return int(stamp.group(1))
    return int(info["resetsAt"]) if info.get("resetsAt") else None


def _session_lost(text: str, stderr: str) -> bool:
    lowered = f"{text} {stderr}".lower()
    return "no conversation found" in lowered or "session not found" in lowered


# --- one turn ---------------------------------------------------------------------------------------

def run(conversation: dict, family: dict, toolbox, library: list[dict],
        family_names: list[str], parts: dict, load_image, installed, *, automatic: bool,
        tools_allowed: bool, fresh: bool = False):
    """One turn: yields its events ({"type": "delta"}, {"type": "tool"}) and returns
    (steps, ends, failure, extra) for conversation_turn to store. fresh: start a new session
    (the old one was lost)."""
    conversation_id = conversation["id"]
    state = {} if fresh else dict(conversation.get("claude_code") or {})
    new_session = not state.get("session_id")
    session_id = str(uuid.uuid4()) if new_session else state["session_id"]
    model = model_for(conversation)
    sections = chat.system_sections(family, workflows=library if toolbox else None,
                                    families=family_names if toolbox else None,
                                    extra_mechanics=CLAUDE_CODE_MECHANICS)
    system = "\n\n".join(sections.values())
    blocks, recorded = build_message(conversation, state, sections, parts, load_image, installed, new_session)

    seat = bot_seat.open_seat(conversation_id, toolbox, tools_allowed=tools_allowed)
    folder = _folder("turns", seat.token)
    files = _write_files(folder, system, seat)
    args = args_for(model, session_id, not new_session, files)
    context = {"mode": "conversation", "conversation_id": conversation_id, "engine": ENGINE}
    if automatic:
        context["automatic"] = True
    record = request_log.RequestRecord({
        "model": f"claude-code/{model}", "session_id": session_id, "resume": not new_session,
        "system": system if new_session else "(recorded when the session started)",
        "messages": [{"role": "user", "content": _logged_blocks(blocks)}],
    }, context)

    steps = [{"text": "", "tool_calls": [], "results": [], "parts": []}]
    texts: list[str] = []  # the reply's text blocks, for the calls' view of it
    calls: dict[str, dict] = {}  # tool_use id -> {"name", "input", "lookup"}
    final, rate_limit, failure, stderr_tail = None, None, None, []
    process = None
    try:
        with _slots:
            process = _spawn(args, environment(), str(_folder("workspace")))
            reader = threading.Thread(target=lambda: stderr_tail.append(process.stderr.read()), daemon=True)
            reader.start()
            timer = threading.Timer(TURN_TIMEOUT_SECONDS, process.kill)
            timer.start()
            try:
                try:
                    process.stdin.write(json.dumps({"type": "user", "message": {"role": "user", "content": blocks}})
                                        + "\n")
                    process.stdin.close()
                except OSError as e:  # it ended before reading the message: what it said, and its exit code, follow
                    logger.warning("claude code closed its input conversation_id=%s: %s", conversation_id, e)
                for line in process.stdout:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        event = json.loads(line)
                    except ValueError:
                        continue
                    kind = event.get("type")
                    if kind == "system" and event.get("subtype") == "init":
                        stopped = _check_tools(event)
                        if stopped:
                            failure = {"detail": stopped}
                            process.kill()
                            break
                    elif kind == "stream_event" and event.get("parent_tool_use_id") is None:
                        delta = (event.get("event") or {}).get("delta") or {}
                        if delta.get("type") == "text_delta" and delta.get("text"):
                            yield {"type": "delta", "text": delta["text"]}
                    elif kind == "assistant" and event.get("parent_tool_use_id") is None:
                        for block in (event.get("message") or {}).get("content") or []:
                            step = steps[-1]
                            if block.get("type") == "text" and block.get("text"):
                                if step["results"]:
                                    steps.append(step := {"text": "", "tool_calls": [], "results": [], "parts": []})
                                step["text"] = f"{step['text']}\n\n{block['text']}" if step["text"] else block["text"]
                                texts.append(block["text"])
                            elif block.get("type") == "tool_use" and str(block.get("name", "")).startswith(TOOL_PREFIX):
                                name, arguments = block["name"][len(TOOL_PREFIX):], block.get("input") or {}
                                lookup = name in bot_seat.LOOKUPS
                                calls[block["id"]] = {"name": name, "input": arguments, "lookup": lookup}
                                seat.saw_call(name, arguments, "\n\n".join(texts))
                                if lookup:
                                    continue
                                if step["results"]:
                                    steps.append(step := {"text": "", "tool_calls": [], "results": [], "parts": []})
                                step["tool_calls"].append({"id": block["id"], "name": name,
                                                           "arguments": json.dumps(arguments, ensure_ascii=False)})
                    elif kind == "user" and event.get("parent_tool_use_id") is None:
                        content = (event.get("message") or {}).get("content")
                        for block in content if isinstance(content, list) else []:
                            call = calls.get(block.get("tool_use_id")) if block.get("type") == "tool_result" else None
                            if call is None or call["lookup"]:
                                continue
                            ran = seat.claim(call["name"], call["input"])
                            if ran is None:  # refused before it reached the app
                                said = block.get("content")
                                said = said if isinstance(said, str) else json.dumps(said, ensure_ascii=False)
                                ran = {"result": {"error": said}, "part": {"type": "tool", "tool": call["name"],
                                                                            "error": said[:200]}}
                            step = next(s for s in steps if any(c["id"] == block["tool_use_id"]
                                                                for c in s["tool_calls"]))
                            step["results"].append(json.dumps(ran["result"], ensure_ascii=False))
                            if ran["part"]:
                                step["parts"].append(ran["part"])
                                yield {"type": "tool", "part": ran["part"]}
                    elif kind == "rate_limit_event":
                        rate_limit = event.get("rate_limit_info")
                    elif kind == "result":
                        final = event
                process.wait(timeout=30)
            finally:
                timer.cancel()
        reader.join(timeout=5)
    except OSError as e:
        failure = {"detail": f"Claude Code couldn't start ({e}). Is it installed? CLAUDE_CODE_COMMAND says how to run it."}
    except subprocess.TimeoutExpired:
        process.kill()
        failure = {"detail": "Claude Code didn't finish."}
    finally:
        bot_seat.close_seat(seat)
        shutil.rmtree(folder, ignore_errors=True)
    stderr = "".join(stderr_tail)

    if failure is None and (final is None or final.get("is_error") or final.get("subtype") != "success"):
        text = str((final or {}).get("result") or "")
        if not new_session and _session_lost(text, stderr) and not any(s["tool_calls"] for s in steps):
            logger.warning("claude code session lost conversation_id=%s session=%s", conversation_id, session_id)
            record.finish(status=None, error="session lost: starting a new one")
            return (yield from run(conversation, family, toolbox, library, family_names, parts,
                                   load_image, installed, automatic=automatic, tools_allowed=tools_allowed,
                                   fresh=True))
        failure = {"detail": _failure_text(final, stderr, process.returncode if process else None)}
        resets = limit_resets(final, rate_limit, stderr)
        if resets:  # the subscription's limit: answered again once it resets (retry_later)
            failure["retry_at"] = resets
    if rate_limit:
        save_limits(rate_limit)
    if failure:
        failure["request_id"] = record.id
        logger.warning("claude code turn failed conversation_id=%s detail=%s", conversation_id, failure["detail"])

    usage = (final or {}).get("usage") or {}
    record.finish(error=(failure or {}).get("detail"), response={
        "text": "\n\n".join(texts), "tool_calls": [{"name": c["name"], "input": c["input"]} for c in calls.values()],
        "subtype": (final or {}).get("subtype"), "stop_reason": (final or {}).get("stop_reason"),
        "num_turns": (final or {}).get("num_turns"), "duration_ms": (final or {}).get("duration_ms"),
        "total_cost_usd": (final or {}).get("total_cost_usd"), "usage": usage, "rate_limit": rate_limit,
        "session_id": (final or {}).get("session_id"),
    })
    if failure is None or any(s["tool_calls"] for s in steps):
        # The session holds this turn now: later turns resume it.
        started = state.get("sections") if not new_session else {n: _digest(sections.get(n)) for n in CHANGING_SECTIONS}
        conversations.set_claude_code(conversation_id, {
            "session_id": session_id, "model": model, "sections": started, **recorded,
            **({"rate_limit": rate_limit} if rate_limit else {}),
        })
    ends = [StreamEnd(cost=None, truncated=(final or {}).get("stop_reason") == "max_tokens", request_id=record.id)]
    extra = {"engine": ENGINE, "claude_code": {
        "model": model, "estimated_cost": (final or {}).get("total_cost_usd"),
        "turns": (final or {}).get("num_turns"), "duration_ms": (final or {}).get("duration_ms")}}
    if rate_limit:
        extra["claude_code"]["rate_limit"] = rate_limit
    return [s for s in steps if s["text"] or s["tool_calls"]] or steps[:1], ends, failure, extra

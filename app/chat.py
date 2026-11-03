"""What the Conversar bot is told, and how its replies are read.

The system prompt is the user's editable personality (the "conversation" system
prompt), then the family rules, characters, numbered notes and (when ComfyUI is
available) the workflow library, and last the fixed mechanics the app depends on --
how to deliver a prompt, how to keep notes, how to title the conversation, how to
run workflows -- placed last and declared to win, so neither the personality nor a
family's rules (written for one-shot modes: "output only the prompt") can break
them. Replies mark prompts, note changes and the title with tags that parse_reply
turns into structured parts; running workflows goes through tools (chat_tools.py).
"""
import json
import re
from dataclasses import dataclass, field
from typing import Callable

from . import image_metadata, prompts, system_prompt

# Every image is re-sent with each request, so only the most recent ones are;
# older ones are replaced by a short note in the message they belonged to.
MAX_CONTEXT_IMAGES = 4
MAX_TITLE_LENGTH = 80
MAX_NOTE_LENGTH = 300

CONVERSATION_MECHANICS = """These mechanics take precedence over anything above that contradicts them, including any instruction about what a reply should contain or look like.

You and the user craft prompts for an AI image model together, in an ongoing conversation. Every prompt you propose must follow the model family rules above.

- Reply in the language the user writes in. Prompts are always written in English.
- When you propose a prompt, put it in its own block, exactly like this:
<prompt>
POSITIVE: <the positive prompt>
NEGATIVE: <the negative prompt, or leave empty if the family doesn't use one>
</prompt>
- The app shows each block as a card the user can copy, numbered across the whole conversation. Your earlier blocks appear as <prompt id="n">, so the user may refer to them by number; never write an id yourself.
- Write your commentary outside the blocks, never inside them. You may propose several alternatives in one reply, one block each, when they are genuinely different.
- Every proposed prompt must be complete and self-contained, ready to paste into the image model.
- Images in the conversation are numbered (Image 1, Image 2...), whether the user shared them or a run made them, so either of you can refer to one by number. An image may come with the metadata it was generated with (prompt, models, LoRAs, settings), shown to you as text next to it: use it. Look at what worked and what didn't, and adjust.
- You can't open links. The one exception is done for you: a link to a Civitai image (civitai.com, .green or .red, /images/<number>) arrives as the image itself, already in the message. A link that didn't arrive that way (a Civitai post or model page, another site) can't be opened: say so and ask for the image's own Civitai link or the image.
- An image the user shares from Civitai comes with what Civitai says about it: prompt, settings and the resources it was made with (checkpoint, LoRAs with weights, base model, trigger words). To reproduce one, match those resources with what the user has installed and say what is missing and what you use instead; to borrow its style, take what defines it (its style words, LoRAs and weights) into the image you are working on, within the family's rules.
- Write plain conversational text: short paragraphs, "- " lists when useful, **bold** sparingly. No headings, no tables, no code blocks.
- Notes: you keep notes about the user across conversations; the ones you have are listed above, numbered. Keep each one telegraphic -- a few words, no full sentences (e.g. "likes: cool dawn light, film grain", "krea-2: LoRA strength >0.6 burns faces"). Change them in your reply, right away and without asking: the user sees every change in the chat and can undo it.
  <remember>note</remember> adds one;
  <remember id="n">note</remember> rewrites note n (to correct or update it, or to merge another into it);
  <forget id="n"/> deletes note n (wrong, outdated, or merged into another).
  Only lasting things -- tastes, what works or fails with a model -- never details of the current image. Rewrite a note rather than add a near-duplicate."""

COMFY_MECHANICS = """- You can generate images on the user's ComfyUI. run_workflow launches a workflow -- one from the library above, "image:N" for the one embedded in image N, or "run:N" for an earlier run's -- with changes such as the prompt, seed, LoRAs, sampler settings, size or any node input. inspect_workflow shows a workflow's nodes and settings; list_comfy_models lists the installed models and LoRAs (use their exact file names); save_workflow keeps a modified workflow in the library, only when the user wants it kept.
- When the user wants to see a prompt rendered or asks for images, run it straight away, without asking first. Show the prompt you run in a <prompt> block in the same reply, unless it already is one of your blocks. Inspect a workflow before structural edits (adding, removing or rewiring nodes).
- Each model belongs to the family its folder is named after (krea-2\\… is Krea 2; the library shows each workflow's family in brackets). A prompt follows the rules of the family of the model it runs on: running a model of another family switches the conversation to that family, and the app hands you its rules to rewrite the prompt by before it runs.
- You work in isolation: the only images you can feed a workflow are this conversation's (with input_images), and only node types from the library, from workflows the user shared here, and common core nodes are available; anything that reads ComfyUI's folders, files or URLs is refused. If a node you need isn't available, say so: the user can add a workflow that has it to the library.
- Runs take a while; their images arrive later, in a message of their own. A message that only reports finished runs is automatic: the user hasn't written. Look at the images, say briefly what worked and what didn't against what was intended, and suggest the next step; don't launch anything on that turn.
- The user can react to the images runs make -- liked, disliked, a comment -- and the reactions come with their next message. Take them as the most direct feedback there is: keep what was liked, fix what wasn't.
- Choose models and LoRAs with what is known about them (installed models above: the user's description, how images made with them went), and note what you learn about how one shapes images, as "<model name>: <effect>"."""

FAMILY_RULES_INTRO = (
    "These rules say how to write the text inside each <prompt> block. Where they describe "
    "a whole reply instead (output only the prompt, no commentary, a single prompt), that "
    "part doesn't apply here: how to reply is set at the end."
)

TITLE_RULE = (
    "- This conversation has no title yet: start your reply with "
    "<title>a 2-6 word title for this conversation, in the user's language</title>."
)

CHARACTERS_INSTRUCTION = (
    "The user is working with these characters. Whenever one appears in a prompt you "
    "propose, describe them exactly as written below, without modifying or paraphrasing "
    "the text, and fill in any unspecified detail consistently."
)

AUTOMATIC_NOTE = "(Automatic message: the user hasn't written.)"
OMITTED_RESULT = {"note": "Result omitted from an earlier turn; call the tool again if you need it."}
# Tool results worth replaying in later turns; the rest (listings) can be fetched again.
KEPT_RESULTS = ("run_workflow", "save_workflow", "cancel_run")
# A family's rules handed over in a tool result matter in that turn; afterwards they are in
# the system prompt (the conversation follows that family now).
FAMILY_RULES_MARK = '"family_rules"'

# A title or note left unclosed ends at the end of its line, not of the reply.
TITLE_PATTERN = re.compile(r"<title>(.*?)(?:</title>|(?=\n)|$)", re.IGNORECASE | re.DOTALL)
REMEMBER_PATTERN = re.compile(
    r"<remember(?:\s+id\s*=\s*[\"']?(\d+)[\"']?)?\s*>(.*?)(?:</remember>|(?=\n)|$)", re.IGNORECASE | re.DOTALL
)
FORGET_PATTERN = re.compile(r"<forget\s+id\s*=\s*[\"']?(\d+)[\"']?\s*/?>(?:\s*</forget>)?", re.IGNORECASE)
# An unclosed block (a reply cut off by the length limit) runs to the end.
PROMPT_PATTERN = re.compile(r"<prompt\b[^>]*>(.*?)(?:</prompt>|$)", re.IGNORECASE | re.DOTALL)
PROMPT_OPEN_PATTERN = re.compile(r"<prompt\b[^>]*>", re.IGNORECASE)
STRAY_TAG_PATTERN = re.compile(r"</?(?:title|remember|forget)\b[^>]*>", re.IGNORECASE)
# A prompt written bare, as "POSITIVE: ..." at the start of a line (the label maybe in **bold**).
BARE_POSITIVE_LINE = re.compile(r"^[ \t]*(?:\*\*)?POSITIVE[ \t]*:", re.IGNORECASE | re.MULTILINE)
BARE_NEGATIVE_START = re.compile(r"^[ \t]*(?:\*\*)?NEGATIVE[ \t]*:", re.IGNORECASE)
BARE_LABEL = re.compile(r"^[ \t]*(?:\*\*)?(POSITIVE|NEGATIVE)[ \t]*:(?:\*\*)?[ \t]*", re.IGNORECASE | re.MULTILINE)
CODE_FENCE_LINE = re.compile(r"^[ \t]*```[\w-]*[ \t]*$\n?", re.MULTILINE)


def build_system_prompt(
    family: dict,
    characters: list[dict],
    notes: list[dict],
    needs_title: bool,
    workflows: list[dict] | None = None,
    models: str | None = None,
) -> str:
    """workflows: the library ({name, description}) when the bot can run ComfyUI
    workflows (None when it can't: no ComfyUI configured, or a model without tools);
    models: the installed models, as catalog.prompt_section() writes them."""
    parts = []
    personality = system_prompt.read_system_prompt("conversation").strip()
    if personality:
        parts.append(personality)
    parts.append(f"## Model Family Rules\n{FAMILY_RULES_INTRO}\n\n{family['instructions'].strip()}")
    if characters:
        lines = "\n".join(f"- {c['name']}: {c['text']}" for c in characters)
        parts.append(f"## Characters\n{CHARACTERS_INSTRUCTION}\n\n{lines}")
    if notes:
        lines = "\n".join(f"[{i}] {n['text']}" for i, n in enumerate(notes, 1))
        parts.append(f"## Your Notes About The User\n{lines}")
    if workflows is not None:
        lines = "\n".join(
            f"- {w['name']}" + (f" [{w['family']}]" if w.get("family") else "")
            + (f": {w['description']}" if w.get("description") else "")
            for w in workflows
        )
        parts.append(f"## Workflow Library\n{lines or '(empty)'}")
        if models:
            parts.append(f"## Installed Models\n{models}")
    mechanics = CONVERSATION_MECHANICS
    if workflows is not None:
        mechanics += f"\n{COMFY_MECHANICS}"
    if needs_title:
        mechanics += f"\n{TITLE_RULE}"
    parts.append(f"## How This Conversation Works\n{mechanics}")
    return "\n\n".join(parts)


def image_numbers(messages: list[dict]) -> dict[str, int]:
    """Every image's number in the conversation: 1, 2, ... in order of appearance."""
    ids = [img["id"] for m in messages for img in (m.get("images") or [])]
    return {image_id: n for n, image_id in enumerate(ids, 1)}


def image_by_number(messages: list[dict], number: int) -> dict | None:
    images = [img for m in messages for img in (m.get("images") or [])]
    return images[number - 1] if 0 < number <= len(images) else None


def _recent_image_ids(messages: list[dict]) -> set[str]:
    ids = [img["id"] for m in messages for img in (m.get("images") or [])]
    return set(ids[-MAX_CONTEXT_IMAGES:])


def results_text(run: dict | None, numbers: dict[str, int], automatic: bool) -> str:
    """A finished run, as the LLM reads it. automatic: nothing came after it yet, so
    the reply being made is the bot's own look at the results."""
    note = f" {AUTOMATIC_NOTE}" if automatic else ""
    if run is None:
        return f"[A run finished.]{note}"
    label = f"Run {run['n']} ({run['workflow']})"
    done = [j for j in run["jobs"] if j.get("images")]
    failed = [j for j in run["jobs"] if j.get("status") in ("error", "interrupted")]
    pieces = []
    for job in done:
        refs = ", ".join(f"image {numbers[i]}" for i in job["images"] if i in numbers)
        pieces.append(f"{refs} (seed {job.get('seed')})" if job.get("seed") is not None else refs)
    if done:
        text = f"[{label} finished: {'; '.join(pieces)}."
    else:
        text = f"[{label} produced no images."
    if failed:
        errors = "; ".join(dict.fromkeys(j.get("error") or j["status"] for j in failed))
        text += f" {len(failed)} of {len(run['jobs'])} job{'s' if len(run['jobs']) > 1 else ''} failed: {errors}."
    return f"{text}]{note}"


def reactions_text(reactions: list[dict], numbers: dict[str, int]) -> str:
    """The user's reactions to images, sent with their message."""
    pieces = []
    for reaction in reactions:
        number = numbers.get(reaction["image_id"])
        if number is None:
            continue
        verdict = {"up": "liked", "down": "disliked"}.get(reaction.get("rating"), "")
        comment = (reaction.get("comment") or "").strip()
        piece = f"image {number}"
        if verdict:
            piece += f" {verdict}"
        if comment:
            piece += f': "{comment}"'
        pieces.append(piece)
    return f"[Reactions: {'; '.join(pieces)}]" if pieces else ""


def _images_content(
    message: dict, numbers: dict[str, int], recent: set[str], load_image, with_metadata: bool
) -> tuple[list[dict], bool]:
    """Text and image parts for a message's images; whether any image is attached."""
    content, attached_any = [], False
    for image in message.get("images") or []:
        number = numbers[image["id"]]
        uri = load_image(image) if image["id"] in recent else None
        label = f"Image {number}" if uri else f"Image {number} (shared earlier, no longer attached)"
        metadata = image.get("metadata") if with_metadata else None
        text = image_metadata.describe(metadata, number) if metadata else ""
        content.append({"type": "text", "text": f"{label}:\n{text}" if text else f"{label}:"})
        if uri:
            content.append({"type": "image_url", "image_url": {"url": uri}})
            attached_any = True
    return content, attached_any


def _flatten(content: list[dict]) -> str | list[dict]:
    """Plain text when there are no images, as before images were numbered."""
    if all(part["type"] == "text" for part in content):
        return "\n\n".join(part["text"] for part in content if part["text"])
    return content


def _tool_message_content(name: str, result_text: str, current: bool) -> str:
    if current or (name in KEPT_RESULTS and FAMILY_RULES_MARK not in result_text):
        return result_text
    return json.dumps(OMITTED_RESULT)


def _steps_messages(steps: list[dict], numbered, structured: bool, current: bool) -> list[dict]:
    """An assistant reply made in steps (text, tool calls, their results) as chat messages."""
    out = []
    for step in steps:
        content = PROMPT_OPEN_PATTERN.sub(numbered, step.get("content") or "")
        calls = step.get("tool_calls") or []
        if structured and calls:
            out.append({
                "role": "assistant",
                "content": content or None,
                "tool_calls": [
                    {"id": c["id"], "type": "function", "function": {"name": c["name"], "arguments": c["arguments"]}}
                    for c in calls
                ],
            })
            for call, result in zip(calls, step.get("results") or []):
                out.append({
                    "role": "tool",
                    "tool_call_id": call["id"],
                    "content": _tool_message_content(call["name"], result, current),
                })
        else:
            if calls:
                done = "; ".join(f"{c['name']}({c['arguments']})" for c in calls)
                content = f"{content}\n\n[Tools used: {done}]".strip()
            if content:
                out.append({"role": "assistant", "content": content})
    return out


def build_llm_messages(
    conversation: dict,
    load_image: Callable[[dict], str | None],
    tools: bool = False,
    current_steps: list[dict] | None = None,
) -> list[dict]:
    """The conversation as chat messages, without the system prompt.

    load_image(image) returns the image as a data URI, or None if it is gone. With
    tools, earlier tool calls are replayed as tool calls (the request must carry the
    tool definitions); without, they are summarized in the text. current_steps are
    the steps of the reply being made, appended last with their results in full.
    """
    messages: list[dict] = []
    all_messages = conversation["messages"]
    recent = _recent_image_ids(all_messages)
    numbers = image_numbers(all_messages)
    runs = {r["n"]: r for r in conversation.get("runs") or []}
    proposal = 0

    def number_block(match: re.Match) -> str:
        nonlocal proposal
        proposal += 1
        return f'<prompt id="{proposal}">'

    def add_user(content: list[dict]) -> None:
        # Consecutive user-side messages (a run's results, then the user) become one.
        if messages and messages[-1]["role"] == "user":
            previous = messages[-1]["content"]
            previous = previous if isinstance(previous, list) else [{"type": "text", "text": previous}]
            messages[-1]["content"] = _flatten(previous + content)
        else:
            messages.append({"role": "user", "content": _flatten(content)})

    for message in all_messages:
        if message["role"] == "assistant":
            steps = message.get("steps") or [{"content": message["content"]}]
            messages += _steps_messages(steps, number_block, tools, current=False)
            continue
        if message["role"] == "results":
            text = results_text(runs.get(message.get("run")), numbers, automatic=message is all_messages[-1])
            images, _ = _images_content(message, numbers, recent, load_image, with_metadata=False)
            add_user([{"type": "text", "text": text}, *images])
            continue
        text = message["content"].strip()
        if message.get("reactions"):
            line = reactions_text(message["reactions"], numbers)
            text = f"{text}\n\n{line}" if text else line
        images, attached = _images_content(message, numbers, recent, load_image, with_metadata=True)
        parts = ([{"type": "text", "text": text}] if text else []) + images
        if not text and not attached and not images:
            parts = [{"type": "text", "text": "(empty message)"}]
        add_user(parts)
    if current_steps:
        messages += _steps_messages(current_steps, number_block, tools, current=True)
    return messages


@dataclass
class Reply:
    content: str  # the reply without notes or title, prompt blocks kept: what the LLM sees next time
    parts: list[dict] = field(default_factory=list)  # {"type": "text", "text"} | {"type": "prompt", ...}
    notes: list[str] = field(default_factory=list)  # new notes
    note_updates: list[tuple[int, str]] = field(default_factory=list)  # (note number, new text)
    note_forgets: list[int] = field(default_factory=list)  # note numbers
    title: str = ""

    @property
    def prompts(self) -> list[dict]:
        return [
            {"positive_prompt": p["positive_prompt"], "negative_prompt": p["negative_prompt"]}
            for p in self.parts
            if p["type"] == "prompt"
        ]


def _wrap_bare_prompts(content: str) -> str:
    """Put a block around each prompt written bare ("POSITIVE: ..." starting a line,
    plus a "NEGATIVE: ..." paragraph right after it), as a model may when it follows a
    family rule like "output only POSITIVE:" over the block format. A bare prompt runs
    to the end of its paragraph, so commentary after a blank line stays commentary."""
    paragraphs = re.split(r"\n[ \t]*\n", content)
    out = []
    i = 0
    while i < len(paragraphs):
        paragraph = paragraphs[i]
        start = BARE_POSITIVE_LINE.search(paragraph)
        if start is None:
            out.append(paragraph)
            i += 1
            continue
        lead = paragraph[:start.start()].rstrip()
        if lead:
            out.append(lead)
        block = paragraph[start.start():]
        next_is_negative = i + 1 < len(paragraphs) and BARE_NEGATIVE_START.match(paragraphs[i + 1])
        if next_is_negative and not re.search(r"NEGATIVE[ \t]*:", block, re.IGNORECASE):
            block += "\n" + paragraphs[i + 1]
            i += 1
        block = BARE_LABEL.sub(lambda m: f"{m.group(1).upper()}: ", block)
        out.append(f"<prompt>\n{block.strip()}\n</prompt>")
        i += 1
    return "\n\n".join(out)


def _note_text(text: str) -> str:
    return " ".join(text.split())[:MAX_NOTE_LENGTH]


def parse_reply(text: str, has_negative_prompt: bool) -> Reply:
    title_match = TITLE_PATTERN.search(text)
    title = " ".join(title_match.group(1).split())[:MAX_TITLE_LENGTH] if title_match else ""
    notes, updates = [], []
    for number, note in REMEMBER_PATTERN.findall(text):
        note = _note_text(note)
        if not note:
            continue
        if number:
            updates.append((int(number), note))
        else:
            notes.append(note)
    forgets = list(dict.fromkeys(int(n) for n in FORGET_PATTERN.findall(text)))

    content = TITLE_PATTERN.sub("", text)
    content = REMEMBER_PATTERN.sub("", content)
    content = FORGET_PATTERN.sub("", content)
    content = STRAY_TAG_PATTERN.sub("", content)
    content = CODE_FENCE_LINE.sub("", content)
    content = re.sub(r"[ \t]+\n", "\n", content)
    content = re.sub(r"\n{3,}", "\n\n", content).strip()
    if not PROMPT_OPEN_PATTERN.search(content):
        # Stored wrapped, so next turn the model sees its proposals in the block format.
        content = _wrap_bare_prompts(content)

    parts = []
    position = 0
    for match in PROMPT_PATTERN.finditer(content):
        before = content[position:match.start()].strip()
        if before:
            parts.append({"type": "text", "text": before})
        positive, negative = prompts.parse_response(match.group(1), has_negative_prompt)
        if positive:
            parts.append({"type": "prompt", "positive_prompt": positive, "negative_prompt": negative})
        position = match.end()
    after = content[position:].strip()
    if after:
        parts.append({"type": "text", "text": after})

    return Reply(
        content=content, parts=parts, notes=notes, note_updates=updates, note_forgets=forgets, title=title
    )

"""What the Conversar bot is told, and how its replies are read.

The system prompt is the user's editable personality (the "conversation" system
prompt), then the family rules, characters and notes, and last the fixed mechanics
the app depends on -- how to deliver a prompt, how to take a note, how to title the
conversation -- placed last and declared to win, so neither the personality nor a
family's rules (written for one-shot modes: "output only the prompt") can break
them. Replies mark prompts, notes and the title with tags that parse_reply turns
into structured parts.
"""
import re
from dataclasses import dataclass, field
from typing import Callable

from . import prompts, system_prompt

# Every image is re-sent with each request, so only the most recent ones are;
# older ones are replaced by a short note in the message they belonged to.
MAX_CONTEXT_IMAGES = 4
MAX_TITLE_LENGTH = 80

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
- The user may share images generated from your prompts, or describe what came out. Look at what worked and what didn't, and adjust.
- Write plain conversational text: short paragraphs, "- " lists when useful, **bold** sparingly. No headings, no tables, no code blocks.
- You keep notes about the user across conversations; any you already have are listed above. When you learn something worth remembering in future conversations -- a lasting taste or preference, or something that works or fails with a model -- add one short note anywhere in your reply as <remember>the note</remember>, right away and without asking first: the user sees every note in the chat and can forget it there. Only lasting things, never details of the current image, and never a note you already have."""

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

NOTES_INTRO = "What you have learned about the user in earlier conversations:"

# A title or note left unclosed ends at the end of its line, not of the reply.
TITLE_PATTERN = re.compile(r"<title>(.*?)(?:</title>|(?=\n)|$)", re.IGNORECASE | re.DOTALL)
REMEMBER_PATTERN = re.compile(r"<remember>(.*?)(?:</remember>|(?=\n)|$)", re.IGNORECASE | re.DOTALL)
# An unclosed block (a reply cut off by the length limit) runs to the end.
PROMPT_PATTERN = re.compile(r"<prompt\b[^>]*>(.*?)(?:</prompt>|$)", re.IGNORECASE | re.DOTALL)
PROMPT_OPEN_PATTERN = re.compile(r"<prompt\b[^>]*>", re.IGNORECASE)
STRAY_TAG_PATTERN = re.compile(r"</?(?:title|remember)>", re.IGNORECASE)
# A prompt written bare, as "POSITIVE: ..." at the start of a line (the label maybe in **bold**).
BARE_POSITIVE_LINE = re.compile(r"^[ \t]*(?:\*\*)?POSITIVE[ \t]*:", re.IGNORECASE | re.MULTILINE)
BARE_NEGATIVE_START = re.compile(r"^[ \t]*(?:\*\*)?NEGATIVE[ \t]*:", re.IGNORECASE)
BARE_LABEL = re.compile(r"^[ \t]*(?:\*\*)?(POSITIVE|NEGATIVE)[ \t]*:(?:\*\*)?[ \t]*", re.IGNORECASE | re.MULTILINE)
CODE_FENCE_LINE = re.compile(r"^[ \t]*```[\w-]*[ \t]*$\n?", re.MULTILINE)


def build_system_prompt(
    family: dict, characters: list[dict], notes: list[dict], needs_title: bool
) -> str:
    parts = []
    personality = system_prompt.read_system_prompt("conversation").strip()
    if personality:
        parts.append(personality)
    parts.append(f"## Model Family Rules\n{FAMILY_RULES_INTRO}\n\n{family['instructions'].strip()}")
    if characters:
        lines = "\n".join(f"- {c['name']}: {c['text']}" for c in characters)
        parts.append(f"## Characters\n{CHARACTERS_INSTRUCTION}\n\n{lines}")
    if notes:
        lines = "\n".join(f"- {n['text']}" for n in notes)
        parts.append(f"## Your Notes About The User\n{NOTES_INTRO}\n{lines}")
    mechanics = CONVERSATION_MECHANICS + (f"\n{TITLE_RULE}" if needs_title else "")
    parts.append(f"## How This Conversation Works\n{mechanics}")
    return "\n\n".join(parts)


def _recent_image_ids(messages: list[dict]) -> set[str]:
    ids = [img["id"] for m in messages for img in (m.get("images") or [])]
    return set(ids[-MAX_CONTEXT_IMAGES:])


def build_llm_messages(
    conversation: dict, load_image: Callable[[dict], str | None]
) -> list[dict]:
    """The conversation as chat messages, without the system prompt.

    load_image(image) returns the image as a data URI, or None if it is gone.
    """
    messages = []
    recent = _recent_image_ids(conversation["messages"])
    proposal = 0

    def number_block(match: re.Match) -> str:
        nonlocal proposal
        proposal += 1
        return f'<prompt id="{proposal}">'

    for message in conversation["messages"]:
        if message["role"] == "assistant":
            content = PROMPT_OPEN_PATTERN.sub(number_block, message["content"])
            messages.append({"role": "assistant", "content": content})
            continue

        text = message["content"].strip()
        images = message.get("images") or []
        attached = [uri for uri in (load_image(img) for img in images if img["id"] in recent) if uri]
        dropped = len(images) - len(attached)
        if dropped:
            note = f"[{dropped} image{'s' if dropped > 1 else ''} shared here earlier, no longer attached]"
            text = f"{text}\n\n{note}" if text else note
        if attached:
            content = [{"type": "text", "text": text or "(image attached)"}]
            content += [{"type": "image_url", "image_url": {"url": uri}} for uri in attached]
        else:
            content = text
        messages.append({"role": "user", "content": content})
    return messages


@dataclass
class Reply:
    content: str  # the reply without notes or title, prompt blocks kept: what the LLM sees next time
    parts: list[dict] = field(default_factory=list)  # {"type": "text", "text"} | {"type": "prompt", ...}
    notes: list[str] = field(default_factory=list)
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


def parse_reply(text: str, has_negative_prompt: bool) -> Reply:
    title_match = TITLE_PATTERN.search(text)
    title = " ".join(title_match.group(1).split())[:MAX_TITLE_LENGTH] if title_match else ""
    notes = [" ".join(m.split()) for m in REMEMBER_PATTERN.findall(text)]
    notes = [n for n in notes if n]

    content = TITLE_PATTERN.sub("", text)
    content = REMEMBER_PATTERN.sub("", content)
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

    return Reply(content=content, parts=parts, notes=notes, title=title)

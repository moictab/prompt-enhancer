"""What the Conversar bot is told, and how its replies are read.

The system prompt is the user's editable personality (the "conversation" system
prompt), then the family rules, numbered notes and (when ComfyUI is
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

# Every image is re-sent with each request, so only recent ones are attached; older ones are
# a short note in the message they belonged to. The window moves in steps, not image by
# image: between steps, what earlier requests sent stays exactly the same, so the
# provider's prompt cache keeps serving it (a window moving with every image would change
# old messages each turn, and the whole conversation would be paid for again). At least
# CONTEXT_IMAGES_MIN, more than a run brings, so the bot sees every image it has to review.
CONTEXT_IMAGES_MIN = 8
CONTEXT_IMAGES_STEP = 8
# Images the user's latest message mentions as #N are attached to it again (up to this many).
MAX_REFERENCED_IMAGES = 4
# "#4": an image's number. Not "#ff0000", "#1a" or a "&#39;" entity.
IMAGE_REF = re.compile(r"(?<![\w&#])#(\d{1,4})\b")
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
- Images in the conversation are numbered (Image 1, Image 2...), whether the user shared them or a run made them, and either of you refers to one as #N (#4 is image 4; the chat turns it into a link to the image): write #N whenever you mention an image. An image the user mentions by #N is attached to their message again, even if it was shared long ago. An image may come with the metadata it was generated with (prompt, models, LoRAs, settings), shown to you as text next to it: use it. Look at what worked and what didn't, and adjust.
- You can't open links. The one exception is done for you: a link to a Civitai image (civitai.com, .green or .red, /images/<number>) arrives as the image itself, already in the message. A link that didn't arrive that way (a Civitai post or model page, another site) can't be opened: say so and ask for the image's own Civitai link or the image.
- An image the user shares from Civitai comes with what Civitai says about it: prompt, settings and the resources it was made with (checkpoint, LoRAs with weights, base model, trigger words). To reproduce one, match those resources with what the user has installed and say what is missing and what you use instead; to borrow its style, take what defines it (its style words, LoRAs and weights) into the image you are working on, within the family's rules.
- Write plain conversational text: short paragraphs, "- " lists when useful, **bold** sparingly. No headings, no tables, no code blocks.
- Notes: you keep notes about the user across conversations; the ones you have come numbered in the app's context at the end of the latest message. Keep each one telegraphic -- a few words, no full sentences (e.g. "likes: cool dawn light, film grain", "krea-2: LoRA strength >0.6 burns faces"). Change them in your reply, right away and without asking: the user sees every change in the chat and can undo it.
  <remember>note</remember> adds one;
  <remember id="n">note</remember> rewrites note n (to correct or update it, or to merge another into it);
  <forget id="n"/> deletes note n (wrong, outdated, or merged into another).
  Only lasting things -- tastes, what works or fails with a model -- never details of the current image. Rewrite a note rather than add a near-duplicate."""

COMFY_MECHANICS = """- You can generate images on the user's ComfyUI. run_workflow launches a workflow -- one from the library above, "image:N" for the one embedded in image N, "run:N" for an earlier run's, or one you write yourself (graph) -- with changes such as the prompt, the base model, seed, LoRAs, sampler settings, size or any node input. inspect_workflow shows a workflow's nodes and settings; describe_nodes shows the node types you can use and what each takes and gives; list_comfy_models lists the installed models and LoRAs (use their exact file names); save_workflow keeps a workflow in the library, only when the user wants it kept.
- Workflows are means, not limits: build what the image needs. Change the model, LoRAs, sampler and size; add, remove or rewire nodes (a hires pass, a detailer, an upscale, an image input); or write a workflow from scratch when none fits. The workflow of an image the user shared is a recipe to learn from and start with, not one to stay in. Check a node type's inputs with describe_nodes before writing or rewiring nodes you haven't used in this conversation.
- Take the initiative: when you write prompts, launch them in the same reply, without asking whether to -- unless the user only wants text, or asked you not to. Show each prompt you run in a <prompt> block in the same reply, before the call, unless it already is one of your blocks. Inspect a workflow before structural edits (adding, removing or rewiring nodes).
- Read the conversation to decide how to work, and say in a few words which way you go and why:
  - Focus when the user is after a particular image: they love one, ask for "this but with…", point at its details, or want a reference reproduced. Make few images (1 to 3), change one or two things at a time, keep the seed and the proportions, so the effect of each change shows.
  - Explore when the request is open ("a witch", "ideas for…", "surprise me"), or when focusing has stalled. Launch a batch of 4 to 8 images that differ from each other on several axes at once: situation and action, setting, time of day, weather and light, pose, framing and camera, wardrobe, mood and palette -- and, where it helps, the base model, the LoRAs and their weights, the style. No two alike and nothing generic: specific, surprising choices, each one an idea of its own. Say in a line what each image tries.
  - Mix them when it fits: part of a batch pushes the direction that works best, part tries new ground.
  Change approach as the conversation does: an image the user loves turns exploring into focusing; rounds that don't improve, or "something else", turn focusing into exploring. Don't repeat a combination already tried: the runs and your reviews say what was.
- Different prompts are different runs, one run_workflow each; count makes several images of one prompt, with different seeds.
- Decide the base model when you write a prompt, not when you launch it: pick it from what is known -- the installed models in the app's context (the user's description of each, how the images made with it went, its past use), your notes, what the user asked -- and say in a few words why when you pick one or change it. Name it in every block: <prompt model="krea-2\\name.safetensors">, the exact file (the library shows each workflow's). That is the model the prompt runs on: when you launch it, the app loads that model in the workflow, whichever it came with (changes.model sets another). The workflow must be able to load it: a checkpoint where it loads a checkpoint, a diffusion model where it loads one, and then of the same family (its text encoders and VAE are that family's).
- A launch that goes through ends your reply: say what you have to say -- what you run and why -- before calling run_workflow or calibrate_lora, since you won't get to add anything after it, nor to call anything else. To launch several runs (variations, alternatives), make all the calls together, in the same step: one run_workflow per run. If a launch fails, you get the error back and can fix the call.
- Choose the image's proportions yourself, for what it shows, without waiting to be told: 2:3 or 3:4 for a portrait or a figure, 9:16 for a full-length figure or anything tall, 1:1 for a face, an object or an icon, 3:2 or 4:3 for a scene, a room or two people side by side, 16:9 for a landscape or a wide scene, 21:9 for a panorama. Set them with the aspect change, which keeps the workflow's pixel count (what its model is made for); change megapixels or give an exact width and height only when the user asks. A workflow's proportions in the library are its default, not a choice made for this image. Decide them at an image's first run and keep them while you refine it (run:N keeps them; a library workflow starts again from its default, so pass aspect again): change them only when the composition changes or the user asks, and say so when you do. The images of one set or story share them, unless a shot calls for others.
- Each model belongs to the family its folder is named after (krea-2\\… is Krea 2; the library shows each workflow's family in brackets). A prompt follows the rules of the family of the model it runs on: running a model of another family switches the conversation to that family, and the app hands you its rules to rewrite the prompt by before it runs.
- The family rules above are the current family's. To write for another family -- the user asks for one, or you'll work with a model of another family -- call use_family first: its rules come back and the conversation keeps that family. If the user asks you to stay with a family, don't run models of other families unless they ask.
- Before telling the user a model or LoRA isn't installed, check with list_comfy_models, searching its file name or the words of its name: files are often renamed, and models are installed all the time. A Civitai image's resources say which installed file each matches, when one does.
- You work in isolation: the only images you can feed a workflow are this conversation's (with input_images), and only node types from the library, from workflows the user shared here, and common core nodes are available; anything that reads ComfyUI's folders, files or URLs is refused. If a node you need isn't available, say so: the user can add a workflow that has it to the library.
- Runs take a while; their images arrive later, in a message of their own. A message that only reports finished runs is automatic: the user hasn't written. Look at the images and say briefly what worked and what didn't against what was intended; then, if you're working on your own, go on yourself (below), and if not, suggest the next step.
- Working on your own: between the user's messages you carry on alone, within an image budget the app's context gives you each turn. After each batch's images, decide the next step yourself (focus or explore) and launch it in the same reply, without waiting for the user, while the budget allows. Stop -- and sum up briefly: what you tried, the best images (#N), what you'd try next -- when the result is there, when you'd only repeat yourself, or when you need something only the user can decide. The user's reactions and words, whenever they come, steer the next batch. For a bigger task with its own goal and limits ("until it looks right", "keep going up to 60 images"), call keep_going with the goal and the user's limits: it replaces the budget between messages until it's done.
- Review every image a run brings, in the reply that first sees it: <review image="N" verdict="ok">…</review> or verdict="flawed", one per image, saying in a few words what's wrong (hands, anatomy, duplicated bodies or faces, garbage text, burnt or washed-out colors, wrong style, not what was asked…) and its likely cause in the recipe (which LoRA or weight, the size, the steps…). Reviews are kept with each image's recipe: the installed models list shows how often each LoRA and weight has failed, and why.
- calibrate_lora shows what a LoRA does and from which weight it breaks: the same prompt and seed at several weights, run when the GPU is free (idle priority, so it can take a while). Review its images weight by weight and note the conclusion as "<lora>: <effect>; fine up to <weight>, breaks at <weight>".
- The user can react to the images runs make -- liked, or disliked with the reason -- and the reactions come with their next message. Take them as the most direct feedback there is, above your own reviews: keep what was liked, fix what wasn't.
- Choose models and LoRAs with what is known about them (the installed models, in the app's context at the end of the latest message: the user's description, how images made with them went), and note what you learn about how one shapes images, as "<model name>: <effect>"."""

FAMILY_RULES_INTRO = (
    "These rules say how to write the text inside each <prompt> block. Where they describe "
    "a whole reply instead (output only the prompt, no commentary, a single prompt), that "
    "part doesn't apply here: how to reply is set at the end."
)

TITLE_RULE = (
    "- This conversation has no title yet: start your reply with "
    "<title>a 2-6 word title for this conversation, in the user's language</title>."
)

COMMENTS_OFF_RULE = (
    "- The user turned off your automatic comments on runs: a run's images reach you with the "
    "user's next message, not on their own -- unless you're working on your own (keep_going). "
    "Otherwise, don't say you'll comment on them when they arrive."
)

LOOKING_RULE = (
    "- This turn is your look at a run's results: nothing can be launched on it (your tools are off). Say "
    "what you'd try next and let the user decide; don't say you're launching it."
)

AUTOMATIC_NOTE = "(Automatic message: the user hasn't written.)"
CONTEXT_HEADING = "[The app's context for this turn, refreshed every turn; not written by the user.]"
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
PROMPT_PATTERN = re.compile(r"<prompt\b([^>]*)>(.*?)(?:</prompt>|$)", re.IGNORECASE | re.DOTALL)
PROMPT_OPEN_PATTERN = re.compile(r"<prompt\b([^>]*)>", re.IGNORECASE)
# The base model a block is written for: <prompt model="krea-2\x.safetensors">.
MODEL_ATTRIBUTE = re.compile(r"""\bmodel\s*=\s*["']([^"'>]*)["']""", re.IGNORECASE)
REVIEW_PATTERN = re.compile(
    r"<review\s+image\s*=\s*[\"']?(\d+)[\"']?(?:\s+verdict\s*=\s*[\"']?(\w+)[\"']?)?\s*>(.*?)(?:</review>|(?=\n)|$)",
    re.IGNORECASE | re.DOTALL,
)
STRAY_TAG_PATTERN = re.compile(r"</?(?:title|remember|forget|review)\b[^>]*>", re.IGNORECASE)
# A prompt written bare, as "POSITIVE: ..." at the start of a line (the label maybe in **bold**).
BARE_POSITIVE_LINE = re.compile(r"^[ \t]*(?:\*\*)?POSITIVE[ \t]*:", re.IGNORECASE | re.MULTILINE)
BARE_NEGATIVE_START = re.compile(r"^[ \t]*(?:\*\*)?NEGATIVE[ \t]*:", re.IGNORECASE)
BARE_LABEL = re.compile(r"^[ \t]*(?:\*\*)?(POSITIVE|NEGATIVE)[ \t]*:(?:\*\*)?[ \t]*", re.IGNORECASE | re.MULTILINE)
CODE_FENCE_LINE = re.compile(r"^[ \t]*```[\w-]*[ \t]*$\n?", re.MULTILINE)


def build_system_prompt(
    family: dict,
    workflows: list[dict] | None = None,
    families: list[str] | None = None,
) -> str:
    """What stays the same from turn to turn (so the provider's prompt cache serves it):
    the personality, the family's rules, the library and the mechanics.
    What changes -- the notes, the installed models with their stats, whether a title is
    still wanted -- goes at the end of the latest message instead (turn_context).

    workflows: the library ({name, description}) when the bot can run ComfyUI
    workflows (None when it can't: no ComfyUI configured, or a model without tools);
    families: the names of the families it can switch to (with use_family)."""
    return "\n\n".join(system_sections(family, workflows, families).values())


def system_sections(
    family: dict,
    workflows: list[dict] | None = None,
    families: list[str] | None = None,
    extra_mechanics: str = "",
) -> dict[str, str]:
    """The system prompt's sections, in order, by name (personality, family, library,
    families, mechanics; the optional ones only when they apply). Claude Code
    (claude_code.py) records its system prompt when a session starts, so it is told
    later which of them changed. extra_mechanics: lines added to the mechanics."""
    sections = {}
    personality = system_prompt.read_system_prompt("conversation").strip()
    if personality:
        sections["personality"] = personality
    heading = f"## Model Family Rules: {family['name']}" if family.get("name") else "## Model Family Rules"
    sections["family"] = f"{heading}\n{FAMILY_RULES_INTRO}\n\n{family['instructions'].strip()}"
    if workflows is not None:
        lines = "\n".join(
            f"- {w['name']}" + (f" [{w['family']}]" if w.get("family") else "")
            + (f" on {w['model']}" if w.get("model") else "")
            + (f" ({w['size']} by default)" if w.get("size") else "")
            + (f": {w['description']}" if w.get("description") else "")
            for w in workflows
        )
        sections["library"] = f"## Workflow Library\n{lines or '(empty)'}"
        others = [name for name in families or [] if name != family.get("name")]
        if others:
            sections["families"] = (f"## Model Families\nThe current one is {family.get('name')}; "
                                    f"the others: {', '.join(others)}.")
    mechanics = CONVERSATION_MECHANICS
    if workflows is not None:
        mechanics += f"\n{COMFY_MECHANICS}"
    if extra_mechanics:
        mechanics += f"\n{extra_mechanics}"
    sections["mechanics"] = f"## How This Conversation Works\n{mechanics}"
    return sections


def autonomy_text(working: dict, automatic: bool, spent: bool) -> str:
    """While the bot works on its own -- within the budget between the user's messages
    (kind "initiative"), or toward a goal it was given (keep_going) --: its budget, and what to do now."""
    used = (f"{working['images']} of {working['max_images']} images made, ${working['cost']:.2f} of "
            f"${working['max_cost']:.2f} spent on your replies")
    if working.get("kind") == "initiative":
        head = f"## Working On Your Own\nBetween the user's messages you carry on alone. Budget since they last wrote: {used}."
        if spent:
            return (f"{head}\nThe budget is spent: don't launch anything. Sum up briefly: what you tried, the best "
                    "images (#N), what you'd do next. The user's next message brings a new budget.")
        if automatic:
            return (f"{head}\nThese are your batch's images. Review them, decide the next step (focus or explore, and "
                    "say which) and launch it in this same reply, without asking -- or don't launch, and sum up, if "
                    "the result is there, you'd only repeat yourself, or you need the user.")
        return (f"{head}\nThe user just wrote: do what they ask, launching what it calls for. Its images come back to "
                "you, and you can go on from there within this budget.")
    head = f"## Working On Your Own\nGoal: {working['goal']}\nBudget used: {used}."
    if spent:
        return (f"{head}\nThe budget is spent: don't launch anything. Sum up for the user: what you tried, the best "
                "images (#N) and what you'd try next.")
    if automatic:
        return (f"{head}\nReview these images against the goal. If it isn't met, decide the next step -- focus on what "
                "most likely failed, or explore if you're stuck -- and launch it in this same reply, without asking. "
                "If it is met, or you're out of good ideas, don't launch: sum up what you tried, the best images (#N) "
                "and what you'd try next.")
    return (f"{head}\nThe user wrote while you work: do what they say. If they want you to stop, call keep_going "
            "with stop: true.")


def turn_context(notes: list[dict], models: str | None = None, needs_title: bool = False,
                 autonomy: str | None = None, comments_off: bool = False, looking: bool = False) -> str | None:
    """What changes from turn to turn, sent at the end of the latest message: the title
    request while the conversation has none, that a run's images won't come back on their
    own (comments_off: the conversation's automatic comments are off), that this turn can't
    launch (looking: the automatic look at a run's results), the bot's goal and budget while it
    works on its own (autonomy_text), the notes (numbered, so the bot can rewrite or forget
    one) and the installed models, as catalog.prompt_section() writes them (with how the
    images made with each went)."""
    sections = [TITLE_RULE] if needs_title else []
    if comments_off:
        sections.append(COMMENTS_OFF_RULE)
    if looking:
        sections.append(LOOKING_RULE)
    if autonomy:
        sections.append(autonomy)
    if notes:
        lines = "\n".join(f"[{i}] {n['text']}" for i, n in enumerate(notes, 1))
        sections.append(f"## Your Notes About The User\n{lines}")
    if models:
        sections.append(f"## Installed Models\n{models}")
    if not sections:
        return None
    return "\n\n".join([CONTEXT_HEADING, *sections])


def image_numbers(messages: list[dict]) -> dict[str, int]:
    """Every image's number in the conversation: 1, 2, ... in order of appearance."""
    ids = [img["id"] for m in messages for img in (m.get("images") or [])]
    return {image_id: n for n, image_id in enumerate(ids, 1)}


def image_by_number(messages: list[dict], number: int) -> dict | None:
    images = [img for m in messages for img in (m.get("images") or [])]
    return images[number - 1] if 0 < number <= len(images) else None


def image_refs(text: str) -> list[int]:
    """The image numbers a text mentions as #N, in order, without repeats."""
    return list(dict.fromkeys(int(n) for n in IMAGE_REF.findall(text or "")))


def _referenced_images(message: dict, all_messages: list[dict], numbers: dict[str, int], recent: set[str],
                       load_image, installed) -> list[dict]:
    """The images a message mentions as #N that aren't attached anyway: attached to it again."""
    own = {img["id"] for img in message.get("images") or []}
    content = []
    for number in image_refs(message.get("content", ""))[:MAX_REFERENCED_IMAGES]:
        image = image_by_number(all_messages, number)
        if image is None or image["id"] in recent or image["id"] in own:
            continue
        uri = load_image(image)
        if not uri:
            continue
        metadata = image.get("metadata")
        text = image_metadata.describe(metadata, number, installed) if metadata else ""
        label = f"Image {number} (mentioned as #{number}, attached again):"
        content += [{"type": "text", "text": f"{label}\n{text}" if text else label},
                    {"type": "image_url", "image_url": {"url": uri}}]
    return content


def _recent_image_ids(messages: list[dict]) -> set[str]:
    """The images attached in full: the latest CONTEXT_IMAGES_MIN to MIN + STEP - 1, the
    window's start moving CONTEXT_IMAGES_STEP at a time."""
    ids = [img["id"] for m in messages for img in (m.get("images") or [])]
    start = max(0, (len(ids) - CONTEXT_IMAGES_MIN) // CONTEXT_IMAGES_STEP * CONTEXT_IMAGES_STEP)
    return set(ids[start:])


def results_text(run: dict | None, numbers: dict[str, int], automatic: bool) -> str:
    """A finished run, as the LLM reads it. automatic: nothing came after it yet, so
    the reply being made is the bot's own look at the results."""
    note = f" {AUTOMATIC_NOTE}" if automatic else ""
    if run is None:
        return f"[A run finished.]{note}"
    calibration = run.get("kind") == "calibration"
    label = f"Run {run['n']} ({run['workflow']}{', ' + run['size'] if run.get('size') else ''})"
    if calibration:
        seed = next((j.get("seed") for j in run["jobs"]), None)
        label = f"Run {run['n']} (calibration of {run.get('lora')} on {run['workflow']}, seed {seed})"
    done = [j for j in run["jobs"] if j.get("images")]
    failed = [j for j in run["jobs"] if j.get("status") in ("error", "interrupted")]
    pieces = []
    for job in done:
        refs = ", ".join(f"image {numbers[i]}" for i in job["images"] if i in numbers)
        if calibration:
            pieces.append(f"{refs} (weight {job.get('calibration_weight'):g})")
        else:
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
    message: dict, numbers: dict[str, int], recent: set[str], load_image, with_metadata: bool,
    installed: dict | None = None,
) -> tuple[list[dict], bool]:
    """Text and image parts for a message's images; whether any image is attached."""
    content, attached_any = [], False
    for image in message.get("images") or []:
        number = numbers[image["id"]]
        uri = load_image(image) if image["id"] in recent else None
        label = f"Image {number}" if uri else f"Image {number} (shared earlier, no longer attached)"
        metadata = image.get("metadata") if with_metadata else None
        text = image_metadata.describe(metadata, number, installed) if metadata else ""
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
    installed: dict | None = None,
    context: str | None = None,
) -> list[dict]:
    """The conversation as chat messages, without the system prompt.

    load_image(image) returns the image as a data URI, or None if it is gone. With
    tools, earlier tool calls are replayed as tool calls (the request must carry the
    tool definitions); without, they are summarized in the text. current_steps are
    the steps of the reply being made, appended last with their results in full.
    context (turn_context) goes as the last part of the latest user-side message.

    What an earlier turn sent is replayed exactly the same, so the prompt cache serves it
    (see conversation_turn.with_cache_breakpoints). Only the end of the latest message
    changes from turn to turn -- the images it mentions as #N, attached again, and the
    context -- and the result's volatile_tail says how many parts that is.
    """
    messages = ChatMessages()
    all_messages = conversation["messages"]
    recent = _recent_image_ids(all_messages)
    numbers = image_numbers(all_messages)
    runs = {r["n"]: r for r in conversation.get("runs") or []}
    tail: list[dict] = []
    proposal = 0
    last_answer = max((i for i, m in enumerate(all_messages) if m["role"] == "assistant"), default=-1)

    def number_block(match: re.Match) -> str:
        nonlocal proposal
        proposal += 1
        model = block_model(match.group(1))
        return f'<prompt id="{proposal}" model="{model}">' if model else f'<prompt id="{proposal}">'

    def add_user(content: list[dict]) -> None:
        # Consecutive user-side messages (a run's results, then the user) become one.
        if messages and messages[-1]["role"] == "user":
            previous = messages[-1]["content"]
            previous = previous if isinstance(previous, list) else [{"type": "text", "text": previous}]
            messages[-1]["content"] = _flatten(previous + content)
        else:
            messages.append({"role": "user", "content": _flatten(content)})

    for index, message in enumerate(all_messages):
        if message["role"] == "assistant":
            steps = message.get("steps") or [{"content": message["content"]}]
            messages += _steps_messages(steps, number_block, tools, current=False)
            continue
        if message["role"] == "results":
            # Automatic: the bot looks (or looked) at it before the user wrote anything.
            following = all_messages[index + 1] if index + 1 < len(all_messages) else None
            automatic = following is None or following["role"] == "assistant"
            text = results_text(runs.get(message.get("run")), numbers, automatic=automatic)
            images, _ = _images_content(message, numbers, recent, load_image, with_metadata=False)
            add_user([{"type": "text", "text": text}, *images])
            continue
        text = message["content"].strip()
        if message.get("balance"):
            # Taking stock (balance.py): the evidence as it was when asked, while it waits for the bot's
            # balance; once answered, a line (tens of KB less in every later turn: the balance follows).
            from . import balance
            asked = message["balance"] if index > last_answer else balance.answered_text(message.get("balance_counts"))
            text = f"{asked}\n\nThe user adds: {text}" if text else asked
        if message.get("reactions"):
            line = reactions_text(message["reactions"], numbers)
            text = f"{text}\n\n{line}" if text else line
        images, attached = _images_content(message, numbers, recent, load_image, with_metadata=True,
                                           installed=installed)
        if message is all_messages[-1]:  # the message being answered
            tail += _referenced_images(message, all_messages, numbers, recent, load_image, installed)
        parts = ([{"type": "text", "text": text}] if text else []) + images
        if not text and not attached and not images:
            parts = [{"type": "text", "text": "(empty message)"}]
        add_user(parts)
    if context:
        tail.append({"type": "text", "text": context})
    if tail and messages and messages[-1]["role"] == "user":
        latest = messages[-1]["content"]
        latest = latest if isinstance(latest, list) else [{"type": "text", "text": latest}]
        messages[-1]["content"] = [*latest, *tail]
        messages.volatile_tail = len(tail)
    if current_steps:
        messages += _steps_messages(current_steps, number_block, tools, current=True)
    return messages


class ChatMessages(list):
    """Chat messages, plus how many parts at the end of the latest user-side message
    change from turn to turn (volatile_tail): the prompt cache's breakpoint goes before them."""
    volatile_tail = 0


@dataclass
class Reply:
    content: str  # the reply without notes or title, prompt blocks kept: what the LLM sees next time
    parts: list[dict] = field(default_factory=list)  # {"type": "text", "text"} | {"type": "prompt", ...}
    notes: list[str] = field(default_factory=list)  # new notes
    note_updates: list[tuple[int, str]] = field(default_factory=list)  # (note number, new text)
    note_forgets: list[int] = field(default_factory=list)  # note numbers
    reviews: list[tuple[int, str, str]] = field(default_factory=list)  # (image number, ok|flawed, note)
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


def block_model(attributes: str) -> str:
    """The model a <prompt> block's attributes name, or ""."""
    match = MODEL_ATTRIBUTE.search(attributes or "")
    return " ".join(match.group(1).split()) if match else ""


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
    reviews = []
    for number, verdict, note in REVIEW_PATTERN.findall(text):
        note = _note_text(note)
        verdict = (verdict or "").lower()
        if verdict not in ("ok", "flawed"):
            verdict = "ok" if note.lower().startswith("ok") else "flawed"
        reviews.append((int(number), verdict, note))

    content = TITLE_PATTERN.sub("", text)
    content = REMEMBER_PATTERN.sub("", content)
    content = FORGET_PATTERN.sub("", content)
    content = REVIEW_PATTERN.sub("", content)
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
        positive, negative = prompts.parse_response(match.group(2), has_negative_prompt)
        if positive:
            part = {"type": "prompt", "positive_prompt": positive, "negative_prompt": negative}
            model = block_model(match.group(1))
            if model:
                part["model"] = model
            parts.append(part)
        position = match.end()
    after = content[position:].strip()
    if after:
        parts.append({"type": "text", "text": after})

    return Reply(
        content=content, parts=parts, notes=notes, note_updates=updates, note_forgets=forgets, reviews=reviews,
        title=title,
    )

import json
from typing import Literal

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, Field

from .. import (away, balance, catalog, chat_tools, civitai, claude_code, comfy, conversation_turn, conversations,
               downloads, history, memory, references, retry_later, runs)
from .api import _require_family, read_image

router = APIRouter(prefix="/api/conversations")

MAX_IMAGES_PER_MESSAGE = 4
MAX_TITLE_LENGTH = 120


def _require_conversation(conversation_id: str) -> dict:
    conversation = conversations.get(conversation_id)
    if conversation is None:
        raise HTTPException(status_code=404, detail="Unknown conversation")
    return conversation


def _require_model(llm_model: str) -> str:
    if not llm_model.strip():
        raise HTTPException(status_code=400, detail="llm_model is required")
    return llm_model.strip()


Engine = Literal["openrouter", "claude-code"]
Effort = Literal["", "low", "medium", "high", "xhigh", "max"]  # "": the model's own (claude_code.EFFORTS)
AutoModel = Literal["", "opus", "sonnet", "haiku"]  # "": the conversation's Claude model (claude_code.MODELS)


class ConversationSettings(BaseModel):
    family_id: str
    llm_model: str
    temperature: float = Field(default=0.7, ge=0, le=2)
    auto_comment: bool = True
    engine: Engine = "openrouter"
    claude_model: str = Field(default="", max_length=80)
    claude_effort: Effort = ""
    claude_auto_model: AutoModel = ""
    image_check: bool = True
    initiative_images: int = Field(default=conversations.INITIATIVE_IMAGES, ge=0, le=chat_tools.INITIATIVE_MAX_IMAGES)


class ConversationSettingsUpdate(BaseModel):
    family_id: str | None = None
    llm_model: str | None = None
    temperature: float | None = Field(default=None, ge=0, le=2)
    title: str | None = Field(default=None, max_length=MAX_TITLE_LENGTH)
    auto_comment: bool | None = None
    engine: Engine | None = None
    claude_model: str | None = Field(default=None, max_length=80)
    claude_effort: Effort | None = None
    claude_auto_model: AutoModel | None = None
    image_check: bool | None = None
    initiative_images: int | None = Field(default=None, ge=0, le=chat_tools.INITIATIVE_MAX_IMAGES)


def _require_idle(conversation_id: str) -> None:
    # One turn at a time per conversation: a message sent meanwhile would land before its reply.
    if conversation_turn.is_busy(conversation_id):
        raise HTTPException(status_code=409, detail="The bot is still replying in this conversation. Wait for it to finish.")


def _with_activity(item: dict) -> dict:
    """Plus what the bot is doing there right now: answering the user, or commenting on results."""
    return {**item, "replying": conversation_turn.is_replying(item["id"]),
            "commenting": conversation_turn.is_commenting(item["id"])}


@router.get("")
def list_conversations():
    return [_with_activity(s) for s in conversations.list_summaries()]


def _require_engine(engine: str | None) -> None:
    if engine == claude_code.ENGINE and not claude_code.available():
        raise HTTPException(status_code=400, detail="Claude Code isn't installed here (CLAUDE_CODE_COMMAND)")


@router.post("")
def create_conversation(req: ConversationSettings):
    _require_family(req.family_id)
    _require_engine(req.engine)
    return conversations.create(
        req.family_id, _require_model(req.llm_model), req.temperature,
        auto_comment=req.auto_comment, engine=req.engine, claude_model=req.claude_model.strip(),
        initiative_images=req.initiative_images, claude_effort=req.claude_effort,
        claude_auto_model=req.claude_auto_model, image_check=req.image_check,
    )


class AwayStart(BaseModel):
    until: float  # when the user is back, epoch seconds
    instructions: str = Field(default="", max_length=2000)
    claude_model: str = Field(default="", max_length=80)
    claude_effort: Effort = ""
    family_id: str | None = None
    # What the night is for: half and half, surprising them (new ideas from random sparks), or their favorites.
    focus: Literal["mix", "explore", "favorites"] = "mix"


@router.post("/away")
def start_away(req: AwayStart):
    """The away mode ("Modo autónomo", away.py): while the user is away, the bot works on its own in a
    new conversation until `until`, on idle GPU and within what Claude Code's 5-hour window can spare.
    Answers the conversation; its first turn is already under way."""
    try:
        conversation = away.start(req.until, req.instructions, req.claude_model.strip(), req.claude_effort,
                                  req.family_id, req.focus)
    except away.AwayError as e:
        raise HTTPException(status_code=409, detail=str(e))
    try:
        conversation_turn.start_turn(conversation["id"])
    except conversation_turn.Busy:
        pass  # can't be: it was just made
    return _with_activity(conversations.get(conversation["id"]) or conversation)


@router.get("/{conversation_id}")
def get_conversation(conversation_id: str):
    """The conversation, plus whether the bot is replying or commenting on results right
    now (the chat then follows the reply, GET …/live). A request to take stock comes without
    the evidence the bot got (tens of KB, which the chat re-reads every few seconds while a
    run is under way): GET …/balance has it."""
    conversation = _require_conversation(conversation_id)
    conversation["messages"] = [{**m, "balance": True} if m.get("balance") else m for m in conversation["messages"]]
    conversation["messages"] = [{**m, "reference_study": True} if m.get("reference_study") else m
                                for m in conversation["messages"]]
    for record in conversation.get("downloads") or []:  # how far the one under way has got
        if record["status"] == "downloading":
            record["progress"] = downloads.progress(conversation_id, record["n"])
    if claude_code.engine(conversation):
        # The subscription's usage as the latest turn reported it (any conversation's: it's the account's).
        conversation["claude_code_limits"] = claude_code.limits()
    return _with_activity(conversation)


@router.patch("/{conversation_id}")
def update_conversation(conversation_id: str, req: ConversationSettingsUpdate):
    changes = req.model_dump(exclude_none=True)
    if "title" in changes:
        changes["title"] = " ".join(changes["title"].split())
        if not changes["title"]:
            raise HTTPException(status_code=400, detail="A title can't be empty")
    if "family_id" in changes:
        _require_family(changes["family_id"])
    if "llm_model" in changes:
        changes["llm_model"] = _require_model(changes["llm_model"])
    _require_engine(changes.get("engine"))
    if "claude_model" in changes:
        changes["claude_model"] = changes["claude_model"].strip()
    updated = conversations.update_settings(conversation_id, changes)
    if updated is None:
        raise HTTPException(status_code=404, detail="Unknown conversation")
    return updated


@router.delete("/{conversation_id}")
def delete_conversation(conversation_id: str):
    # Its pending runs would only keep ComfyUI busy for images nobody will see.
    for run in (conversations.get(conversation_id) or {}).get("runs") or []:
        if run["status"] == "pending":
            runs.cancel(conversation_id, run["n"])
    if not conversations.delete(conversation_id):
        raise HTTPException(status_code=404, detail="Unknown conversation")
    return {"deleted": True}


@router.get("/{conversation_id}/images/{image_id}")
def get_image(conversation_id: str, image_id: str, small: bool = False):
    found = conversations.image_file(conversation_id, image_id, small=small)
    if found is None:
        raise HTTPException(status_code=404, detail="Unknown image")
    path, content_type = found
    return FileResponse(path, media_type=content_type)


class Reaction(BaseModel):
    rating: Literal["up", "love", "down"] | None = None  # love: a favorite, the strongest like
    comment: str = ""
    # Why it was disliked, picked with a tap: hands, face, anatomy, style, composition, not_asked.
    reasons: list[Literal["hands", "face", "anatomy", "style", "composition", "not_asked"]] = Field(
        default_factory=list, max_length=6)


@router.put("/{conversation_id}/images/{image_id}/reaction")
def react_to_image(conversation_id: str, image_id: str, reaction: Reaction):
    """Like, dislike or comment on an image; it reaches the bot with the next message."""
    _require_conversation(conversation_id)
    if reaction.rating == "down" and not reaction.comment.strip() and not reaction.reasons:
        # A dislike is only worth learning from with its reason.
        raise HTTPException(status_code=400, detail="A thumbs down needs its reason: say what went wrong")
    image = conversations.set_reaction(conversation_id, image_id, reaction.rating, reaction.comment,
                                       reasons=reaction.reasons)
    if image is None:
        raise HTTPException(status_code=404, detail="Unknown image")
    catalog.invalidate_stats()
    return image


@router.post("/{conversation_id}/messages")
async def send_message(
    conversation_id: str,
    content: str = Form(""),
    images: list[UploadFile] | None = File(None),
):
    conversation = _require_conversation(conversation_id)
    _require_idle(conversation_id)
    images = images or []
    if not content.strip() and not images and not conversations.unsent_reactions(conversation):
        raise HTTPException(status_code=400, detail="Write a message or attach an image")
    civitai_ids = civitai.image_ids(content)
    if len(images) + len(civitai_ids) > MAX_IMAGES_PER_MESSAGE:
        raise HTTPException(
            status_code=400,
            detail=f"At most {MAX_IMAGES_PER_MESSAGE} images per message (Civitai links included)",
        )
    _require_family(conversation["family_id"])

    # Validate every image before storing any, so a bad one leaves nothing behind.
    contents = [await read_image(image) for image in images]
    shared = []
    for image_id in civitai_ids:
        try:
            shared.append(await run_in_threadpool(civitai.fetch_image, image_id))
        except civitai.CivitaiError as e:
            raise HTTPException(status_code=400, detail=str(e))
    # An accepted reference folder ("@name", or its path): the app studies it for the bot.
    study = None
    mention = references.find_mention(content, references.list_folders())
    if mention:
        try:
            study = await run_in_threadpool(references.study, *mention, catalog.load()["entries"] or None)
        except references.ReferenceFolderError as e:
            raise HTTPException(status_code=400, detail=str(e))
    # Reading metadata and making the small copy takes a moment: off the event loop.
    saved = [
        await run_in_threadpool(conversations.save_image, conversation_id, data, image.content_type)
        for image, data in zip(images, contents)
    ] + [
        await run_in_threadpool(conversations.save_image, conversation_id, item.data, item.content_type,
                                None, item.workflow, item.metadata)
        for item in shared
    ]
    message = {"role": "user", "content": content.strip(), "images": saved}
    if study:
        sheets = [await run_in_threadpool(conversations.save_image, conversation_id, data, "image/jpeg")
                  for data in study.sheets]
        message["images"] = saved + [{**sheet, "sheet": True} for sheet in sheets]
        message["references"] = study.counts
        message["reference_study"] = study.text
    away.user_active()  # they're back: the away mode, wherever it works, ends
    conversations.append_message(conversation_id, message)
    _start_initiative(conversation_id)
    return _reply(conversation_id)


def _start_initiative(conversation_id: str) -> None:
    """The user wrote: until their next message the bot may carry on alone, within the
    conversation's image budget (none without ComfyUI, or with the budget at 0)."""
    conversation = conversations.get(conversation_id) or {}
    images = conversations.initiative(conversation)
    if images > 0 and comfy.client() is not None:
        conversations.start_initiative(conversation_id, images, chat_tools.INITIATIVE_COST)
    elif (conversations.autonomy(conversation) or {}).get("kind") == "initiative":
        conversations.end_autonomy(conversation_id, "user")


@router.post("/{conversation_id}/reply")
def retry_reply(conversation_id: str):
    """Answer the last message again, after a reply failed or was cut off (or a
    run's results went uncommented)."""
    conversation = _require_conversation(conversation_id)
    if not conversation["messages"] or conversation["messages"][-1]["role"] not in ("user", "results"):
        raise HTTPException(status_code=409, detail="The last message already has a reply")
    _require_idle(conversation_id)
    _require_family(conversation["family_id"])
    away.user_active()
    return _reply(conversation_id)


class DownloadChoice(BaseModel):
    folder: str | None = Field(default=None, max_length=40)
    family: str | None = Field(default=None, max_length=80)
    name: str | None = Field(default=None, max_length=140)


def _download_action(action, conversation_id: str, n: int, *args):
    _require_conversation(conversation_id)
    try:
        return action(conversation_id, n, *args)
    except LookupError:
        raise HTTPException(status_code=404, detail="Unknown download")
    except downloads.DownloadError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/{conversation_id}/downloads/{n}/accept")
def accept_download(conversation_id: str, n: int, choice: DownloadChoice):
    """The user accepts a download the bot prepared, with the folder, family folder and name they chose."""
    return _download_action(downloads.accept, conversation_id, n, choice.folder, choice.family, choice.name)


@router.post("/{conversation_id}/downloads/{n}/decline")
def decline_download(conversation_id: str, n: int):
    return _download_action(downloads.decline, conversation_id, n)


@router.post("/{conversation_id}/downloads/{n}/cancel")
def cancel_download(conversation_id: str, n: int):
    _require_conversation(conversation_id)
    downloads.cancel(conversation_id, n)
    return {"cancelling": True}


@router.delete("/{conversation_id}/retry")
def cancel_retry(conversation_id: str):
    """Don't answer the last message again by itself when Claude Code's limit resets."""
    _require_conversation(conversation_id)
    away.user_active()
    retry_later.cancel(conversation_id)
    conversations.set_retry(conversation_id, None)
    return {"retry_at": None}


@router.post("/{conversation_id}/redo")
def redo_reply(conversation_id: str):
    """Discard the bot's last reply and have it answer again ("Rehacer"), told it is answering
    again: the runs it launched that are still pending are cancelled and their images won't
    come, the notes it took are forgotten and those it changed or forgot restored, and its
    history entry goes. The budget to carry on alone starts again, as after a message."""
    conversation = _require_conversation(conversation_id)
    if not conversation["messages"] or conversation["messages"][-1]["role"] != "assistant":
        raise HTTPException(status_code=409, detail="The last message isn't the bot's reply")
    _require_idle(conversation_id)
    working = conversations.autonomy(conversation)
    if working and working.get("active") and working.get("kind") != "initiative":
        raise HTTPException(status_code=409, detail="The bot is working on its own here: stop it first")
    away.user_active()
    _require_family(conversation["family_id"])
    try:
        # Held so that none of its runs can put its images in the chat in between.
        with conversation_turn.turn_lock(conversation_id):
            taken = conversations.discard_last_reply(conversation_id)
            if taken is None:
                raise HTTPException(status_code=409, detail="The last message isn't the bot's reply")
            reply = taken[0]
            launched = [p["run"] for p in reply.get("parts") or [] if p.get("type") == "run"]
            for n in launched:
                conversations.update_run(conversation_id, n, lambda run: run.update(discarded=True))
    except conversation_turn.Busy as e:
        raise HTTPException(status_code=409, detail=str(e))
    for n in launched:
        if (conversations.get_run(conversation_id, n) or {}).get("status") == "pending":
            runs.cancel(conversation_id, n)
    for note in reply.get("notes") or []:
        memory.delete_note(note["id"])
    for change in reversed(reply.get("note_changes") or []):
        if change["action"] == "update":
            memory.update_note(change["id"], change["previous"])
        elif change["action"] == "forget":
            memory.create_note(change["text"], conversation_id)
    if reply.get("history_id"):
        history.delete_entry(reply["history_id"])
    if not (working and working.get("active") and working.get("kind") != "initiative"):
        _start_initiative(conversation_id)
    return _reply(conversation_id)


MAX_FOCUS_LENGTH = 2000


@router.get("/{conversation_id}/balance")
def balance_data(conversation_id: str):
    """The evidence for taking stock of the conversation (balance.py), without asking the bot:
    every image its runs made with its recipe, review and reaction, and the counts per ingredient."""
    found = balance.ledger(_require_conversation(conversation_id))
    if found is None:
        raise HTTPException(status_code=404, detail="The runs here have made no images yet")
    return {"text": found.text, **found.counts}


@router.post("/{conversation_id}/finish")
def finish_conversation(conversation_id: str, focus: str = Form("")):
    """End the conversation (writing again reopens it). When its runs made images, the bot takes
    stock of it first: what worked best and worst, and its notes brought in line with that -- a
    message of its own (the evidence, and whatever the user adds as focus), answered like any
    other but without tools, streamed the same way. With nothing to take stock of, the
    conversation is only marked: the answer is the conversation, as JSON."""
    _require_conversation(conversation_id)
    _require_idle(conversation_id)
    away.user_active()  # they're back; and the away mode's own conversation can be ended like any other
    conversation = _require_conversation(conversation_id)
    working = conversations.autonomy(conversation)
    if working and working.get("kind") != "initiative":
        raise HTTPException(status_code=409, detail="The bot is working on its own here: stop it before ending")
    if working:  # what it does between messages ends with the conversation
        conversations.end_autonomy(conversation_id, "user")
    if len(focus) > MAX_FOCUS_LENGTH:
        raise HTTPException(status_code=400, detail=f"Keep what you add under {MAX_FOCUS_LENGTH} characters")
    found = balance.ledger(conversation)
    if found is None:
        return conversations.finish(conversation_id)
    _require_family(conversation["family_id"])
    conversations.append_message(conversation_id, {
        "role": "user", "content": focus.strip(), "images": [], "balance": balance.request_text(found),
        "balance_counts": found.counts,
    })
    return _reply(conversation_id)


@router.get("/{conversation_id}/live")
def follow_reply(conversation_id: str):
    """The reply being written there right now, from its start, as the same NDJSON events:
    for a chat that didn't ask for it (another tab, a reload, the bot's comment on results).
    204 when there is none."""
    _require_conversation(conversation_id)
    live = conversation_turn.live_turn(conversation_id)
    if live is None:
        return Response(status_code=204)
    return _follow(live)


@router.post("/{conversation_id}/runs/{n}/cancel")
def cancel_run(conversation_id: str, n: int):
    _require_conversation(conversation_id)
    run = conversations.get_run(conversation_id, n)
    if run is None:
        raise HTTPException(status_code=404, detail="Unknown run")
    if run["status"] != "pending":
        raise HTTPException(status_code=409, detail="The run has already finished")
    # Cancelling is stopping: the bot doesn't carry on alone after it.
    away.user_active()
    conversations.end_autonomy(conversation_id, "user")
    return runs.cancel(conversation_id, n)


@router.post("/{conversation_id}/runs/{n}/prioritize")
def prioritize_run(conversation_id: str, n: int):
    """"Pasar delante": the run's images still waiting in the queue go ahead of everything else
    waiting, other conversations' and other projects' (runs.prioritize)."""
    _require_conversation(conversation_id)
    run = conversations.get_run(conversation_id, n)
    if run is None:
        raise HTTPException(status_code=404, detail="Unknown run")
    if run["status"] != "pending":
        raise HTTPException(status_code=409, detail="The run has already finished")
    if comfy.client() is None:
        raise HTTPException(status_code=503, detail="ComfyUI isn't configured")
    result = runs.prioritize(conversation_id, n)
    if not result["raised"] and result["errors"]:
        raise HTTPException(status_code=502, detail=f"The gateway didn't take it: {result['errors'][0]}")
    return result


@router.post("/{conversation_id}/autonomy/stop")
def stop_autonomy(conversation_id: str):
    """The bot stops working on its own: it launches nothing more, and comments on whatever
    is still running when it arrives, as usual (away, it lands without a comment)."""
    conversation = _require_conversation(conversation_id)
    if away.is_away(conversations.autonomy(conversation)):
        return away.stop(conversation_id, "user")
    ended = conversations.end_autonomy(conversation_id, "user")
    if ended is None:
        raise HTTPException(status_code=409, detail="The bot isn't working on its own here")
    return ended


@router.post("/{conversation_id}/runs/{n}/retry-downloads")
def retry_run_downloads(conversation_id: str, n: int):
    """Fetch again the images of a run whose download failed, or whose jobs were given up for their
    time and finished later (they are still in ComfyUI)."""
    _require_conversation(conversation_id)
    run = conversations.get_run(conversation_id, n)
    if run is None:
        raise HTTPException(status_code=404, detail="Unknown run")
    retried = runs.retry_downloads(conversation_id, n)
    if retried is None:
        raise HTTPException(status_code=409, detail="Nothing of this run to fetch: ComfyUI has no images of it")
    return retried


@router.get("/{conversation_id}/runs/{n}/workflow")
def run_workflow_file(conversation_id: str, n: int, job: int = 1):
    """The exact workflow sent to ComfyUI for a run's job, to load it there."""
    _require_conversation(conversation_id)
    sent = conversations.run_workflows(conversation_id, n)
    if not 0 < job <= len(sent):
        raise HTTPException(status_code=404, detail="Unknown run")
    return JSONResponse(
        sent[job - 1],
        headers={"Content-Disposition": f'attachment; filename="conversar-run-{n}-{job}.json"'},
    )


def _event(payload: dict) -> str:
    return json.dumps(payload, ensure_ascii=False) + "\n"


def _reply(conversation_id: str) -> StreamingResponse:
    """Start the bot's turn and stream it as NDJSON events (see conversation_turn): the HTTP
    status is already 200 by the time anything can fail, so failures travel as events. The
    turn doesn't depend on this response: if the chat goes away, it goes on and is stored."""
    try:
        live = conversation_turn.start_turn(conversation_id)
    except conversation_turn.Busy as e:
        raise HTTPException(status_code=409, detail=str(e))
    return _follow(live)


def _follow(live: conversation_turn.LiveTurn) -> StreamingResponse:
    def events():
        for event in live.follow():
            yield "\n" if event is None else _event(event)  # a blank line keeps a quiet connection open

    return StreamingResponse(events(), media_type="application/x-ndjson")

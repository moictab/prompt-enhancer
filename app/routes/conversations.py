import json
from typing import Literal

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, Field

from .. import balance, catalog, civitai, conversation_turn, conversations, runs
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


class ConversationSettings(BaseModel):
    family_id: str
    llm_model: str
    temperature: float = Field(default=0.7, ge=0, le=2)
    character_ids: list[str] = []
    auto_comment: bool = True


class ConversationSettingsUpdate(BaseModel):
    family_id: str | None = None
    llm_model: str | None = None
    temperature: float | None = Field(default=None, ge=0, le=2)
    character_ids: list[str] | None = None
    title: str | None = Field(default=None, max_length=MAX_TITLE_LENGTH)
    auto_comment: bool | None = None


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


@router.post("")
def create_conversation(req: ConversationSettings):
    _require_family(req.family_id)
    return conversations.create(
        req.family_id, _require_model(req.llm_model), req.temperature, req.character_ids,
        auto_comment=req.auto_comment,
    )


@router.get("/{conversation_id}")
def get_conversation(conversation_id: str):
    """The conversation, plus whether the bot is replying or commenting on results right
    now (the chat then follows the reply, GET …/live). A request to take stock comes without
    the evidence the bot got (tens of KB, which the chat re-reads every few seconds while a
    run is under way): GET …/balance has it."""
    conversation = _require_conversation(conversation_id)
    conversation["messages"] = [{**m, "balance": True} if m.get("balance") else m for m in conversation["messages"]]
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
    rating: Literal["up", "down"] | None = None
    comment: str = ""


@router.put("/{conversation_id}/images/{image_id}/reaction")
def react_to_image(conversation_id: str, image_id: str, reaction: Reaction):
    """Like, dislike or comment on an image; it reaches the bot with the next message."""
    _require_conversation(conversation_id)
    if reaction.rating == "down" and not reaction.comment.strip():
        # A dislike is only worth learning from with its reason.
        raise HTTPException(status_code=400, detail="A thumbs down needs its reason: say what went wrong")
    image = conversations.set_reaction(conversation_id, image_id, reaction.rating, reaction.comment)
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
    # Reading metadata and making the small copy takes a moment: off the event loop.
    saved = [
        await run_in_threadpool(conversations.save_image, conversation_id, data, image.content_type)
        for image, data in zip(images, contents)
    ] + [
        await run_in_threadpool(conversations.save_image, conversation_id, item.data, item.content_type,
                                None, item.workflow, item.metadata)
        for item in shared
    ]
    conversations.append_message(
        conversation_id, {"role": "user", "content": content.strip(), "images": saved}
    )
    return _reply(conversation_id)


@router.post("/{conversation_id}/reply")
def retry_reply(conversation_id: str):
    """Answer the last message again, after a reply failed or was cut off (or a
    run's results went uncommented)."""
    conversation = _require_conversation(conversation_id)
    if not conversation["messages"] or conversation["messages"][-1]["role"] not in ("user", "results"):
        raise HTTPException(status_code=409, detail="The last message already has a reply")
    _require_idle(conversation_id)
    _require_family(conversation["family_id"])
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


@router.post("/{conversation_id}/balance")
def take_stock(conversation_id: str, focus: str = Form("")):
    """Ask the bot to take stock of the whole conversation: what worked best and worst, and its
    notes brought in line with that. A message of its own (with the evidence, and whatever the
    user adds as focus), answered like any other, without tools; streamed the same way."""
    conversation = _require_conversation(conversation_id)
    _require_idle(conversation_id)
    if conversations.autonomy(conversation):
        raise HTTPException(status_code=409, detail="The bot is working on its own here: stop it before taking stock")
    _require_family(conversation["family_id"])
    if len(focus) > MAX_FOCUS_LENGTH:
        raise HTTPException(status_code=400, detail=f"Keep what you add under {MAX_FOCUS_LENGTH} characters")
    found = balance.ledger(conversation)
    if found is None:
        raise HTTPException(status_code=400, detail="There is nothing to take stock of yet: the runs here have made no images")
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
    conversations.end_autonomy(conversation_id, "user")
    return runs.cancel(conversation_id, n)


@router.post("/{conversation_id}/autonomy/stop")
def stop_autonomy(conversation_id: str):
    """The bot stops working on its own: it launches nothing more, and comments on whatever
    is still running when it arrives, as usual."""
    _require_conversation(conversation_id)
    ended = conversations.end_autonomy(conversation_id, "user")
    if ended is None:
        raise HTTPException(status_code=409, detail="The bot isn't working on its own here")
    return ended


@router.post("/{conversation_id}/runs/{n}/retry-downloads")
def retry_run_downloads(conversation_id: str, n: int):
    """Fetch again the images of a run whose download failed (they are still in ComfyUI)."""
    _require_conversation(conversation_id)
    run = conversations.get_run(conversation_id, n)
    if run is None:
        raise HTTPException(status_code=404, detail="Unknown run")
    retried = runs.retry_downloads(conversation_id, n)
    if retried is None:
        raise HTTPException(status_code=409, detail="The run has no failed downloads to retry")
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

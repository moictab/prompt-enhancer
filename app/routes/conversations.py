import json
from typing import Literal

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from .. import catalog, conversation_turn, conversations, runs
from .api import _require_family, read_image

router = APIRouter(prefix="/api/conversations")

MAX_IMAGES_PER_MESSAGE = 4


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


class ConversationSettingsUpdate(BaseModel):
    family_id: str | None = None
    llm_model: str | None = None
    temperature: float | None = Field(default=None, ge=0, le=2)
    character_ids: list[str] | None = None


@router.get("")
def list_conversations():
    return conversations.list_summaries()


@router.post("")
def create_conversation(req: ConversationSettings):
    _require_family(req.family_id)
    return conversations.create(
        req.family_id, _require_model(req.llm_model), req.temperature, req.character_ids
    )


@router.get("/{conversation_id}")
def get_conversation(conversation_id: str):
    """The conversation, plus whether the bot is commenting on results right now
    (the chat polls while that or a run is pending)."""
    conversation = _require_conversation(conversation_id)
    return {**conversation, "commenting": conversation_turn.is_commenting(conversation_id)}


@router.patch("/{conversation_id}")
def update_conversation(conversation_id: str, req: ConversationSettingsUpdate):
    changes = req.model_dump(exclude_none=True)
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
    images = images or []
    if not content.strip() and not images and not conversations.unsent_reactions(conversation):
        raise HTTPException(status_code=400, detail="Write a message or attach an image")
    if len(images) > MAX_IMAGES_PER_MESSAGE:
        raise HTTPException(
            status_code=400, detail=f"At most {MAX_IMAGES_PER_MESSAGE} images per message"
        )
    _require_family(conversation["family_id"])

    # Validate every image before storing any, so a bad one leaves nothing behind.
    contents = [await read_image(image) for image in images]
    # Reading metadata and making the small copy takes a moment: off the event loop.
    saved = [
        await run_in_threadpool(conversations.save_image, conversation_id, data, image.content_type)
        for image, data in zip(images, contents)
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
    _require_family(conversation["family_id"])
    return _reply(conversation_id)


@router.post("/{conversation_id}/runs/{n}/cancel")
def cancel_run(conversation_id: str, n: int):
    _require_conversation(conversation_id)
    run = conversations.get_run(conversation_id, n)
    if run is None:
        raise HTTPException(status_code=404, detail="Unknown run")
    if run["status"] != "pending":
        raise HTTPException(status_code=409, detail="The run has already finished")
    return runs.cancel(conversation_id, n)


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
    """Stream the bot's turn as NDJSON events (see conversation_turn): the HTTP status
    is already 200 by the time anything can fail, so failures travel as events."""
    def events():
        for event in conversation_turn.run_turn(conversation_id):
            yield _event(event)

    return StreamingResponse(events(), media_type="application/x-ndjson")

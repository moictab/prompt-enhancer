import json
import logging
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from .. import chat, conversations, history, memory
from ..config import get_settings
from ..openrouter_client import StreamEnd, stream_chat
from .api import _character_params, _require_family, _selected_characters, data_uri, read_image

router = APIRouter(prefix="/api/conversations")
logger = logging.getLogger("app.conversations")

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
    return _require_conversation(conversation_id)


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
    if not conversations.delete(conversation_id):
        raise HTTPException(status_code=404, detail="Unknown conversation")
    return {"deleted": True}


@router.get("/{conversation_id}/images/{image_id}")
def get_image(conversation_id: str, image_id: str):
    found = conversations.image_file(conversation_id, image_id)
    if found is None:
        raise HTTPException(status_code=404, detail="Unknown image")
    path, content_type = found
    return FileResponse(path, media_type=content_type)


@router.post("/{conversation_id}/messages")
async def send_message(
    conversation_id: str,
    content: str = Form(""),
    images: list[UploadFile] | None = File(None),
):
    conversation = _require_conversation(conversation_id)
    images = images or []
    if not content.strip() and not images:
        raise HTTPException(status_code=400, detail="Write a message or attach an image")
    if len(images) > MAX_IMAGES_PER_MESSAGE:
        raise HTTPException(
            status_code=400, detail=f"At most {MAX_IMAGES_PER_MESSAGE} images per message"
        )
    family = _require_family(conversation["family_id"])

    # Validate every image before storing any, so a bad one leaves nothing behind.
    contents = [await read_image(image) for image in images]
    saved = [
        conversations.save_image(conversation_id, data, image.content_type)
        for image, data in zip(images, contents)
    ]
    conversations.append_message(
        conversation_id, {"role": "user", "content": content.strip(), "images": saved}
    )
    return _reply(conversation_id, family)


@router.post("/{conversation_id}/reply")
def retry_reply(conversation_id: str):
    """Answer the last message again, after a reply failed or was cut off."""
    conversation = _require_conversation(conversation_id)
    if not conversation["messages"] or conversation["messages"][-1]["role"] != "user":
        raise HTTPException(status_code=409, detail="The last message already has a reply")
    return _reply(conversation_id, _require_family(conversation["family_id"]))


def _event(payload: dict) -> str:
    return json.dumps(payload, ensure_ascii=False) + "\n"


def _image_loader(conversation_id: str):
    def load(image: dict) -> str | None:
        found = conversations.image_file(conversation_id, image["id"])
        if found is None:
            return None
        path, content_type = found
        return data_uri(content_type, Path(path).read_bytes())

    return load


def _reply(conversation_id: str, family: dict) -> StreamingResponse:
    """Stream the bot's reply as NDJSON events: {"type": "delta", "text"} while it
    arrives, then {"type": "done", "message", "conversation"} once it is stored, or
    {"type": "error", "detail"} -- the HTTP status is already 200 by then."""
    conversation = conversations.get(conversation_id)
    selected_characters = _selected_characters(conversation["character_ids"])
    system = chat.build_system_prompt(
        family, selected_characters, memory.list_notes(), needs_title=not conversation["title"]
    )
    messages = [
        {"role": "system", "content": system},
        *chat.build_llm_messages(conversation, _image_loader(conversation_id)),
    ]
    model = conversation["llm_model"]
    temperature = conversation["temperature"]
    api_key = get_settings().openrouter_api_key
    logger.info(
        "conversation reply conversation_id=%s family_id=%s model=%s messages=%d",
        conversation_id, family["id"], model, len(messages) - 1,
    )

    def events():
        chunks = []
        end = StreamEnd(cost=None)
        try:
            for item in stream_chat(api_key, model, messages, temperature):
                if isinstance(item, StreamEnd):
                    end = item
                else:
                    chunks.append(item)
                    yield _event({"type": "delta", "text": item})
        except RuntimeError as e:
            logger.warning("conversation reply failed conversation_id=%s error=%s", conversation_id, e)
            yield _event({"type": "error", "detail": str(e)})
            return
        if not "".join(chunks).strip():
            yield _event({"type": "error", "detail": "The model returned an empty reply. Try again."})
            return
        updated = _store_reply(conversation_id, family, selected_characters, "".join(chunks), end)
        if updated is None:
            yield _event({"type": "error", "detail": "The conversation was deleted while replying."})
            return
        yield _event({
            "type": "done",
            "message": updated["messages"][-1],
            "conversation": conversations.summary(updated),
        })

    return StreamingResponse(events(), media_type="application/x-ndjson")


def _store_reply(
    conversation_id: str, family: dict, selected_characters: list[dict], text: str, end: StreamEnd
) -> dict | None:
    """Save the finished reply, its notes, and (when it proposes prompts) a history entry."""
    conversation = conversations.get(conversation_id)
    if conversation is None:
        return None
    reply = chat.parse_reply(text, family["has_negative_prompt"])
    notes = [memory.create_note(note, conversation_id) for note in reply.notes]

    history_id = None
    if reply.prompts:
        last_user = next(m for m in reversed(conversation["messages"]) if m["role"] == "user")
        entry = history.append_entry(
            mode="conversation",
            family_id=family["id"],
            family_name=family["name"],
            llm_model=conversation["llm_model"],
            vision_model=None,
            temperature=conversation["temperature"],
            user_input=last_user["content"],
            example_prompts="",
            previous_prompt=None,
            prompts=reply.prompts,
            cost=end.cost,
            params={
                "conversation_id": conversation_id,
                "conversation_title": reply.title or conversations.display_title(conversation),
                **_character_params(selected_characters),
            },
        )
        history_id = entry["id"]

    logger.info(
        "conversation reply stored conversation_id=%s prompts=%d notes=%d cost=%s truncated=%s",
        conversation_id, len(reply.prompts), len(notes), end.cost, end.truncated,
    )
    return conversations.append_message(
        conversation_id,
        {
            "role": "assistant",
            "content": reply.content,
            "parts": reply.parts,
            "notes": [{"id": n["id"], "text": n["text"]} for n in notes],
            "cost": end.cost,
            "truncated": end.truncated,
            "history_id": history_id,
        },
        title=reply.title,
    )

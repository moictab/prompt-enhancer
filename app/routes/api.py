"""What the pages share: the model list, the history, and the checks for families and uploaded images."""
from fastapi import APIRouter, HTTPException, UploadFile
from pydantic import BaseModel

from .. import families, history
from ..openrouter_client import list_models

router = APIRouter(prefix="/api")

ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp"}
MAX_IMAGE_BYTES = 10 * 1024 * 1024


def _require_family(family_id: str) -> dict:
    family = families.get_family(family_id)
    if family is None:
        raise HTTPException(status_code=404, detail="Unknown family_id")
    return family


async def read_image(image: UploadFile) -> bytes:
    """The uploaded image's bytes, after checking its type and size (400 otherwise)."""
    if image.content_type not in ALLOWED_IMAGE_TYPES:
        raise HTTPException(
            status_code=400, detail=f"Unsupported image type: {image.content_type}"
        )
    contents = await image.read()
    if len(contents) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=400, detail="Image exceeds 10 MB limit")
    return contents


@router.get("/openrouter-models")
def get_openrouter_models():
    try:
        return list_models()
    except RuntimeError as e:
        raise HTTPException(status_code=502, detail=str(e))


@router.get("/model-folders")
def get_model_folders():
    """The family folders under ComfyUI's models folders, where a download from Civitai may go
    (empty when downloads aren't set up: COMFY_MODELS_DIR)."""
    from .. import downloads
    return downloads.family_folders()


@router.get("/history")
def get_history():
    return history.list_entries()


class HistoryEntryUpdate(BaseModel):
    favorite: bool


@router.patch("/history/{entry_id}")
def update_history_entry(entry_id: str, payload: HistoryEntryUpdate):
    entry = history.set_favorite(entry_id, payload.favorite)
    if entry is None:
        raise HTTPException(status_code=404, detail="Unknown history entry")
    return entry


@router.delete("/history/{entry_id}")
def delete_history_entry(entry_id: str):
    if not history.delete_entry(entry_id):
        raise HTTPException(status_code=404, detail="Unknown history entry")
    return {"deleted": True}

import requests
from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel

from .. import catalog, characters, comfy, families, image_metadata, memory, system_prompt, workflows
from ..seed import DEFAULT_SYSTEM_PROMPTS
from .api import read_image

router = APIRouter(prefix="/api/admin")


class FamilyPayload(BaseModel):
    name: str
    instructions: str
    has_negative_prompt: bool


@router.get("/families")
def list_families():
    return families.list_families()


@router.post("/families")
def create_family(payload: FamilyPayload):
    return families.create_family(payload.name, payload.instructions, payload.has_negative_prompt)


@router.put("/families/{family_id}")
def update_family(family_id: str, payload: FamilyPayload):
    updated = families.update_family(
        family_id, payload.name, payload.instructions, payload.has_negative_prompt
    )
    if updated is None:
        raise HTTPException(status_code=404, detail="Unknown family_id")
    return updated


@router.delete("/families/{family_id}")
def delete_family(family_id: str):
    if not families.delete_family(family_id):
        raise HTTPException(status_code=404, detail="Unknown family_id")
    return {"deleted": True}


class CharacterPayload(BaseModel):
    name: str
    text: str


@router.get("/characters")
def list_characters():
    return characters.list_characters()


@router.post("/characters")
def create_character(payload: CharacterPayload):
    return characters.create_character(payload.name, payload.text)


@router.put("/characters/{character_id}")
def update_character(character_id: str, payload: CharacterPayload):
    updated = characters.update_character(character_id, payload.name, payload.text)
    if updated is None:
        raise HTTPException(status_code=404, detail="Unknown character_id")
    return updated


@router.delete("/characters/{character_id}")
def delete_character(character_id: str):
    if not characters.delete_character(character_id):
        raise HTTPException(status_code=404, detail="Unknown character_id")
    return {"deleted": True}


class NotePayload(BaseModel):
    text: str


def _require_note_text(payload: NotePayload) -> str:
    if not payload.text.strip():
        raise HTTPException(status_code=400, detail="text is required")
    return payload.text.strip()


@router.get("/memory")
def list_notes():
    return memory.list_notes()


@router.post("/memory")
def create_note(payload: NotePayload):
    return memory.create_note(_require_note_text(payload))


@router.put("/memory/{note_id}")
def update_note(note_id: str, payload: NotePayload):
    updated = memory.update_note(note_id, _require_note_text(payload))
    if updated is None:
        raise HTTPException(status_code=404, detail="Unknown note")
    return updated


@router.delete("/memory/{note_id}")
def delete_note(note_id: str):
    if not memory.delete_note(note_id):
        raise HTTPException(status_code=404, detail="Unknown note")
    return {"deleted": True}


class SystemPromptPayload(BaseModel):
    text: str


VALID_SYSTEM_PROMPT_MODES = set(DEFAULT_SYSTEM_PROMPTS)


def _require_valid_mode(mode: str) -> None:
    if mode not in VALID_SYSTEM_PROMPT_MODES:
        raise HTTPException(status_code=400, detail=f"Unknown mode: {mode}")


@router.get("/system-prompt/{mode}")
def get_system_prompt(mode: str):
    _require_valid_mode(mode)
    return {"text": system_prompt.read_system_prompt(mode)}


@router.get("/system-prompt/{mode}/default")
def get_default_system_prompt(mode: str):
    _require_valid_mode(mode)
    return {"text": DEFAULT_SYSTEM_PROMPTS[mode]}


@router.put("/system-prompt/{mode}")
def update_system_prompt(mode: str, payload: SystemPromptPayload):
    _require_valid_mode(mode)
    system_prompt.write_system_prompt(mode, payload.text)
    return {"text": payload.text}


class WorkflowPayload(BaseModel):
    name: str
    description: str = ""
    workflow: dict


@router.get("/workflows")
def list_workflows():
    return workflows.list_workflows()


@router.post("/workflows")
def create_workflow(payload: WorkflowPayload):
    try:
        return workflows.create(payload.name, payload.description, payload.workflow)
    except workflows.WorkflowError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.put("/workflows/{workflow_id}")
def update_workflow(workflow_id: str, payload: WorkflowPayload):
    try:
        updated = workflows.update(workflow_id, payload.name, payload.description, payload.workflow)
    except workflows.WorkflowError as e:
        raise HTTPException(status_code=400, detail=str(e))
    if updated is None:
        raise HTTPException(status_code=404, detail="Unknown workflow")
    return updated


@router.delete("/workflows/{workflow_id}")
def delete_workflow(workflow_id: str):
    if not workflows.delete(workflow_id):
        raise HTTPException(status_code=404, detail="Unknown workflow")
    return {"deleted": True}


@router.post("/workflows/extract")
async def extract_workflow(image: UploadFile = File(...)):
    """The API-format workflow embedded in an image generated by ComfyUI."""
    metadata, workflow = image_metadata.read(await read_image(image))
    if workflow is None:
        detail = "The image has no ComfyUI workflow"
        if (metadata or {}).get("editor_workflow_only"):
            detail += " in API format (only the editor graph)"
        raise HTTPException(status_code=400, detail=detail)
    return {"workflow": workflow, "metadata": metadata}


def _catalog_view() -> dict:
    data = catalog.load()
    file_stats = catalog.stats()
    items = [
        {"file": file, "folder": entry["folder"], "family": entry.get("family", ""),
         **catalog.details(file, data, file_stats)}
        for file, entry in sorted(data["entries"].items())
    ]
    return {"refreshed_at": data["refreshed_at"], "configured": comfy.client() is not None, "items": items,
            "history_source": data["history_source"]}


@router.get("/catalog")
def get_catalog():
    return _catalog_view()


@router.post("/catalog/refresh")
def refresh_catalog():
    client = comfy.client()
    if client is None:
        raise HTTPException(status_code=400, detail="ComfyUI isn't configured (COMFY_GATEWAY_URL)")
    try:
        catalog.refresh(client)
    except (comfy.Unreachable, comfy.ComfyError) as e:
        raise HTTPException(status_code=502, detail=str(e))
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=502, detail=f"ComfyUI didn't answer in time: {e.__class__.__name__}")
    return _catalog_view()


class DescriptionPayload(BaseModel):
    file: str
    text: str = ""


@router.put("/catalog/description")
def set_model_description(payload: DescriptionPayload):
    """The owner's description of what a model or LoRA does, for the bot."""
    catalog.set_description(payload.file, payload.text)
    return catalog.details(payload.file)

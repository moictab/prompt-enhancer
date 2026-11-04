import requests
from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel

from .. import catalog, comfy, families, image_metadata, memory, memory_tidy, references, system_prompt, workflows
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


class NotePayload(BaseModel):
    text: str
    # Where the note goes (memory.py): general when left out; "family" with family (a model folder),
    # "topic" with topic. Left out on a rewrite, the note keeps its scope.
    scope: str | None = None
    family: str | None = None
    topic: str | None = None


def _require_note_text(payload: NotePayload) -> str:
    if not payload.text.strip():
        raise HTTPException(status_code=400, detail="text is required")
    return payload.text.strip()


def _scope(payload: NotePayload) -> dict | None:
    if payload.scope is None and not payload.family and not payload.topic:
        return None
    if payload.scope not in (None, *memory.SCOPES):
        raise HTTPException(status_code=400, detail=f"scope is one of {', '.join(memory.SCOPES)}")
    if payload.scope == "family" and not (payload.family or "").strip():
        raise HTTPException(status_code=400, detail="A family note needs its family (the model folder)")
    if payload.scope == "topic" and not (payload.topic or "").strip():
        raise HTTPException(status_code=400, detail="A topic note needs its topic")
    return memory.clean_scope(payload.scope, payload.family, payload.topic)


@router.get("/memory")
def list_notes():
    return memory.list_notes()


@router.post("/memory")
def create_note(payload: NotePayload):
    return memory.create_note(_require_note_text(payload), scope=_scope(payload))


@router.put("/memory/{note_id}")
def update_note(note_id: str, payload: NotePayload):
    updated = memory.update_note(note_id, _require_note_text(payload), scope=_scope(payload))
    if updated is None:
        raise HTTPException(status_code=404, detail="Unknown note")
    return updated


class TidyRequest(BaseModel):
    engine: str = "claude-code"  # or "openrouter"
    model: str = ""  # Claude's (sonnet, opus...) or an OpenRouter model; empty: the default


class TidyApply(BaseModel):
    accept: list[int]  # the plan's operations to apply, by index


@router.get("/memory/tidy")
def tidy_state():
    """The proposal waiting for the user (or null), and whether the last applied one can be undone."""
    return {"plan": memory_tidy.pending(), "can_undo": memory_tidy.can_undo()}


@router.post("/memory/tidy")
def tidy_propose(payload: TidyRequest):
    """Ordenar la memoria: the bot reviews every note and proposes changes, applied only as picked."""
    if payload.engine not in ("claude-code", "openrouter"):
        raise HTTPException(status_code=400, detail="engine is claude-code or openrouter")
    try:
        return memory_tidy.propose(payload.engine, payload.model.strip())
    except memory_tidy.TidyError as e:
        raise HTTPException(status_code=502, detail=str(e))


@router.post("/memory/tidy/apply")
def tidy_apply(payload: TidyApply):
    try:
        return memory_tidy.apply(payload.accept)
    except memory_tidy.TidyError as e:
        raise HTTPException(status_code=409, detail=str(e))


@router.delete("/memory/tidy")
def tidy_discard():
    memory_tidy.discard()
    return {"discarded": True}


@router.post("/memory/tidy/undo")
def tidy_undo():
    try:
        return memory_tidy.undo()
    except memory_tidy.TidyError as e:
        raise HTTPException(status_code=409, detail=str(e))


class TopicKeywordsPayload(BaseModel):
    keywords: list[str]


@router.get("/memory/topics")
def list_topics():
    """The topics the notes have, each with its keywords besides its name (they bring its notes into
    a conversation whose latest messages or title mention them)."""
    keywords = memory.topic_keywords()
    return [{"name": name, "keywords": keywords.get(name, [])} for name in memory.topic_names()]


@router.put("/memory/topics/{name}")
def set_topic_keywords(name: str, payload: TopicKeywordsPayload):
    if name not in memory.topic_names():
        raise HTTPException(status_code=404, detail="No note has that topic")
    return {"name": name, "keywords": memory.set_topic_keywords(name, payload.keywords)}


@router.delete("/memory/{note_id}")
def delete_note(note_id: str):
    if not memory.delete_note(note_id):
        raise HTTPException(status_code=404, detail="Unknown note")
    return {"deleted": True}


class ReferenceFolderPayload(BaseModel):
    name: str
    path: str


@router.get("/reference-folders")
def list_reference_folders():
    """The folders a message may point the bot at, with how many images each has now."""
    return [references.with_counts(entry) for entry in references.list_folders()]


@router.post("/reference-folders")
def create_reference_folder(payload: ReferenceFolderPayload):
    try:
        return references.with_counts(references.create_folder(payload.name, payload.path))
    except references.ReferenceFolderError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.delete("/reference-folders/{folder_id}")
def delete_reference_folder(folder_id: str):
    if not references.delete_folder(folder_id):
        raise HTTPException(status_code=404, detail="Unknown reference folder")
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

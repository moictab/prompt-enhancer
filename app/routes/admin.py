from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from .. import characters, families, system_prompt

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


class SystemPromptPayload(BaseModel):
    text: str


VALID_SYSTEM_PROMPT_MODES = {"generate", "iterate", "image"}


@router.get("/system-prompt/{mode}")
def get_system_prompt(mode: str):
    if mode not in VALID_SYSTEM_PROMPT_MODES:
        raise HTTPException(status_code=400, detail=f"Unknown mode: {mode}")
    return {"text": system_prompt.read_system_prompt(mode)}


@router.put("/system-prompt/{mode}")
def update_system_prompt(mode: str, payload: SystemPromptPayload):
    if mode not in VALID_SYSTEM_PROMPT_MODES:
        raise HTTPException(status_code=400, detail=f"Unknown mode: {mode}")
    system_prompt.write_system_prompt(mode, payload.text)
    return {"text": payload.text}

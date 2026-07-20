import base64

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from .. import characters, families, history, prompts
from ..config import get_settings
from ..openrouter_client import call_openrouter

router = APIRouter(prefix="/api")


class GenerateRequest(BaseModel):
    user_input: str
    family_id: str
    llm_model: str
    temperature: float = 0.7
    example_prompts: str = ""
    previous_prompt: str = ""


@router.post("/generate")
def generate(req: GenerateRequest):
    if not req.user_input.strip():
        raise HTTPException(status_code=400, detail="user_input is required")

    family = families.get_family(req.family_id)
    if family is None:
        raise HTTPException(status_code=404, detail="Unknown family_id")

    is_iteration = bool(req.previous_prompt.strip())
    mode = "iterate" if is_iteration else "generate"
    previous_prompt = req.previous_prompt if is_iteration else None

    settings = get_settings()
    system = prompts.build_system_prompt(mode, family)
    user_message = prompts.build_user_message(
        mode, req.user_input,
        previous_prompt=previous_prompt,
        example_prompts=req.example_prompts,
    )

    try:
        response = call_openrouter(
            api_key=settings.openrouter_api_key,
            model=req.llm_model,
            system_prompt=system,
            user_message=user_message,
            temperature=req.temperature,
        )
    except RuntimeError as e:
        raise HTTPException(status_code=502, detail=str(e))

    positive, negative = prompts.parse_response(response, family["has_negative_prompt"])

    history.append_entry(
        mode=mode,
        family_id=family["id"],
        family_name=family["name"],
        llm_model=req.llm_model,
        vision_model=None,
        temperature=req.temperature,
        user_input=req.user_input,
        example_prompts=req.example_prompts,
        previous_prompt=previous_prompt,
        positive_prompt=positive,
        negative_prompt=negative,
    )

    return {"positive_prompt": positive, "negative_prompt": negative}


ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp"}
MAX_IMAGE_BYTES = 10 * 1024 * 1024


@router.post("/from-image")
async def from_image(
    image: UploadFile = File(...),
    family_id: str = Form(...),
    vision_model: str = Form(...),
    user_input: str = Form(""),
    example_prompts: str = Form(""),
    temperature: float = Form(0.7),
):
    if image.content_type not in ALLOWED_IMAGE_TYPES:
        raise HTTPException(
            status_code=400, detail=f"Unsupported image type: {image.content_type}"
        )

    contents = await image.read()
    if len(contents) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=400, detail="Image exceeds 10 MB limit")

    family = families.get_family(family_id)
    if family is None:
        raise HTTPException(status_code=404, detail="Unknown family_id")

    image_data_uri = f"data:{image.content_type};base64,{base64.b64encode(contents).decode('ascii')}"

    settings = get_settings()
    system = prompts.build_system_prompt("image", family)
    user_message = prompts.build_user_message(
        "image", user_input, example_prompts=example_prompts
    )

    try:
        response = call_openrouter(
            api_key=settings.openrouter_api_key,
            model=vision_model,
            system_prompt=system,
            user_message=user_message,
            temperature=temperature,
            image_data_uri=image_data_uri,
        )
    except RuntimeError as e:
        raise HTTPException(status_code=502, detail=str(e))

    positive, negative = prompts.parse_response(response, family["has_negative_prompt"])

    history.append_entry(
        mode="image",
        family_id=family["id"],
        family_name=family["name"],
        llm_model=None,
        vision_model=vision_model,
        temperature=temperature,
        user_input=user_input,
        example_prompts=example_prompts,
        previous_prompt=None,
        positive_prompt=positive,
        negative_prompt=negative,
    )

    return {"positive_prompt": positive, "negative_prompt": negative}


@router.get("/history")
def get_history():
    return history.list_entries()


@router.get("/characters")
def get_characters():
    return characters.list_characters()

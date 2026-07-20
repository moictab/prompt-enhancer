from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from .. import families, history, prompts, system_prompt
from ..config import get_settings
from ..openrouter_client import call_openrouter

router = APIRouter(prefix="/api")


class GenerateRequest(BaseModel):
    user_input: str
    family_id: str
    llm_model: str
    temperature: float = 0.7
    example_prompts: str = ""


class IterateRequest(GenerateRequest):
    previous_prompt: str


@router.post("/generate")
def generate(req: GenerateRequest):
    if not req.user_input.strip():
        raise HTTPException(status_code=400, detail="user_input is required")

    family = families.get_family(req.family_id)
    if family is None:
        raise HTTPException(status_code=404, detail="Unknown family_id")

    settings = get_settings()
    system = prompts.build_system_prompt(
        system_prompt.read_system_prompt(), family, is_iteration=False
    )
    user_message = prompts.build_user_message(
        "generate", req.user_input, example_prompts=req.example_prompts
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
        mode="generate",
        family_id=family["id"],
        family_name=family["name"],
        llm_model=req.llm_model,
        vision_model=None,
        temperature=req.temperature,
        user_input=req.user_input,
        example_prompts=req.example_prompts,
        previous_prompt=None,
        positive_prompt=positive,
        negative_prompt=negative,
    )

    return {"positive_prompt": positive, "negative_prompt": negative}


@router.post("/iterate")
def iterate(req: IterateRequest):
    if not req.user_input.strip():
        raise HTTPException(status_code=400, detail="user_input is required")
    if not req.previous_prompt.strip():
        raise HTTPException(status_code=400, detail="previous_prompt is required")

    family = families.get_family(req.family_id)
    if family is None:
        raise HTTPException(status_code=404, detail="Unknown family_id")

    settings = get_settings()
    system = prompts.build_system_prompt(
        system_prompt.read_system_prompt(), family, is_iteration=True
    )
    user_message = prompts.build_user_message(
        "iterate", req.user_input,
        previous_prompt=req.previous_prompt, example_prompts=req.example_prompts,
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
        mode="iterate",
        family_id=family["id"],
        family_name=family["name"],
        llm_model=req.llm_model,
        vision_model=None,
        temperature=req.temperature,
        user_input=req.user_input,
        example_prompts=req.example_prompts,
        previous_prompt=req.previous_prompt,
        positive_prompt=positive,
        negative_prompt=negative,
    )

    return {"positive_prompt": positive, "negative_prompt": negative}

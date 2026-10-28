import base64
import logging
from concurrent.futures import ThreadPoolExecutor

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

from .. import characters, families, history, prompts, system_prompt
from ..config import get_settings
from ..openrouter_client import OpenRouterResult, call_openrouter, list_models

router = APIRouter(prefix="/api")
logger = logging.getLogger("app.api")

ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp"}
MAX_IMAGE_BYTES = 10 * 1024 * 1024
MAX_VARIANTS = 4
MAX_REFERENCE_PROMPTS = 12


def _require_family(family_id: str) -> dict:
    family = families.get_family(family_id)
    if family is None:
        raise HTTPException(status_code=404, detail="Unknown family_id")
    return family


def _selected_characters(character_ids: list[str]) -> list[dict]:
    return [c for c in characters.list_characters() if c["id"] in character_ids]


async def _read_image_data_uri(image: UploadFile) -> tuple[str, int]:
    if image.content_type not in ALLOWED_IMAGE_TYPES:
        raise HTTPException(
            status_code=400, detail=f"Unsupported image type: {image.content_type}"
        )
    contents = await image.read()
    if len(contents) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=400, detail="Image exceeds 10 MB limit")
    encoded = base64.b64encode(contents).decode("ascii")
    return f"data:{image.content_type};base64,{encoded}", len(contents)


def _call_llm(
    label: str,
    model: str,
    system: str,
    user_messages: list[str],
    temperature: float,
    image_data_uri: str | None = None,
) -> list[OpenRouterResult]:
    """Run one OpenRouter call per user message, in parallel when there are several.

    Variants are independent samples, so a failed one is dropped rather than
    failing the request; only when every call fails does this raise a 502.
    """
    api_key = get_settings().openrouter_api_key

    def attempt(user_message: str) -> OpenRouterResult | RuntimeError:
        try:
            return call_openrouter(
                api_key=api_key,
                model=model,
                system_prompt=system,
                user_message=user_message,
                temperature=temperature,
                image_data_uri=image_data_uri,
            )
        except RuntimeError as e:
            logger.warning("%s request failed model=%s error=%s", label, model, e)
            return e

    if len(user_messages) == 1:
        outcomes = [attempt(user_messages[0])]
    else:
        with ThreadPoolExecutor(max_workers=len(user_messages)) as pool:
            outcomes = list(pool.map(attempt, user_messages))

    results = [o for o in outcomes if isinstance(o, OpenRouterResult)]
    if not results:
        raise HTTPException(status_code=502, detail=str(outcomes[0]))
    return results


def _total_cost(results: list[OpenRouterResult]) -> float | None:
    costs = [r.cost for r in results if r.cost is not None]
    return sum(costs) if costs else None


def _record(
    mode: str,
    family: dict,
    results: list[OpenRouterResult],
    pairs: list[tuple[str, str]],
    expected_count: int,
    temperature: float,
    user_input: str,
    example_prompts: str = "",
    llm_model: str | None = None,
    vision_model: str | None = None,
    previous_prompt: str | None = None,
    params: dict | None = None,
) -> dict:
    """Log the generation to history and build the response every prompt-crafting
    endpoint returns: the history entry itself plus per-request warning flags."""
    entry = history.append_entry(
        mode=mode,
        family_id=family["id"],
        family_name=family["name"],
        llm_model=llm_model,
        vision_model=vision_model,
        temperature=temperature,
        user_input=user_input,
        example_prompts=example_prompts,
        previous_prompt=previous_prompt,
        prompts=[
            {"positive_prompt": positive, "negative_prompt": negative}
            for positive, negative in pairs
        ],
        cost=_total_cost(results),
        params=params,
    )
    count_mismatch = len(pairs) != expected_count
    logger.info(
        "%s request succeeded family_id=%s prompts=%d expected=%d cost=%s",
        mode, family["id"], len(pairs), expected_count, entry["cost"],
    )
    if count_mismatch:
        logger.warning(
            "%s request produced a different number of prompts than requested "
            "family_id=%s expected=%d got=%d",
            mode, family["id"], expected_count, len(pairs),
        )
    return {
        **entry,
        "truncated": any(r.truncated for r in results),
        "count_mismatch": count_mismatch,
    }


def _character_params(selected_characters: list[dict]) -> dict:
    return {"character_names": [c["name"] for c in selected_characters]} if selected_characters else {}


class GenerateRequest(BaseModel):
    user_input: str
    family_id: str
    llm_model: str
    temperature: float = 0.7
    example_prompts: str = ""
    previous_prompt: str = ""
    character_ids: list[str] = []
    variants: int = Field(default=1, ge=1, le=MAX_VARIANTS)


@router.post("/generate")
def generate(req: GenerateRequest):
    if not req.user_input.strip():
        raise HTTPException(status_code=400, detail="user_input is required")

    family = _require_family(req.family_id)

    is_iteration = bool(req.previous_prompt.strip())
    mode = "iterate" if is_iteration else "generate"
    previous_prompt = req.previous_prompt if is_iteration else None
    selected_characters = _selected_characters(req.character_ids)

    system = prompts.build_system_prompt(mode, family)
    user_messages = [
        prompts.build_user_message(
            mode, req.user_input,
            previous_prompt=previous_prompt,
            example_prompts=req.example_prompts,
            characters=selected_characters,
            variant=(index, req.variants),
        )
        for index in range(1, req.variants + 1)
    ]

    logger.info(
        "generate request mode=%s family_id=%s model=%s temperature=%.2f characters=%d variants=%d",
        mode, req.family_id, req.llm_model, req.temperature, len(selected_characters), req.variants,
    )
    results = _call_llm(mode, req.llm_model, system, user_messages, req.temperature)
    pairs = [prompts.parse_response(r.content, family["has_negative_prompt"]) for r in results]

    return _record(
        mode, family, results, pairs,
        expected_count=req.variants,
        temperature=req.temperature,
        user_input=req.user_input,
        example_prompts=req.example_prompts,
        llm_model=req.llm_model,
        previous_prompt=previous_prompt,
        params={"variants": req.variants, **_character_params(selected_characters)},
    )


class StoryRequest(BaseModel):
    user_input: str
    family_id: str
    llm_model: str
    temperature: float = 0.7
    example_prompts: str = ""
    character_ids: list[str] = []
    num_episodes: int = Field(default=4, ge=2, le=12)


@router.post("/story")
def story(req: StoryRequest):
    if not req.user_input.strip():
        raise HTTPException(status_code=400, detail="user_input is required")

    family = _require_family(req.family_id)
    selected_characters = _selected_characters(req.character_ids)

    system = prompts.build_system_prompt("story", family)
    user_message = prompts.build_user_message(
        "story", req.user_input,
        example_prompts=req.example_prompts,
        characters=selected_characters,
        count=req.num_episodes,
    )

    logger.info(
        "story request family_id=%s model=%s temperature=%.2f characters=%d num_episodes=%d",
        req.family_id, req.llm_model, req.temperature, len(selected_characters), req.num_episodes,
    )
    results = _call_llm("story", req.llm_model, system, [user_message], req.temperature)
    pairs = prompts.parse_multi_response(results[0].content, family["has_negative_prompt"])

    return _record(
        "story", family, results, pairs,
        expected_count=req.num_episodes,
        temperature=req.temperature,
        user_input=req.user_input,
        example_prompts=req.example_prompts,
        llm_model=req.llm_model,
        params={"num_episodes": req.num_episodes, **_character_params(selected_characters)},
    )


class VariationAxis(BaseModel):
    key: str
    vary: bool
    value: str = ""


class VariationsRequest(BaseModel):
    family_id: str
    llm_model: str
    axes: list[VariationAxis]
    user_input: str = ""
    aesthetic: str = ""
    temperature: float = 0.7
    example_prompts: str = ""
    character_ids: list[str] = []
    count: int = Field(default=4, ge=2, le=12)
    # Prompts the user already likes (usually of the same subject) for the LLM to
    # synthesize the constant base from.
    reference_prompts: list[str] = Field(default=[], max_length=MAX_REFERENCE_PROMPTS)


@router.post("/variations")
def variations(req: VariationsRequest):
    unknown = [a.key for a in req.axes if a.key not in prompts.VARIATION_AXES]
    if unknown:
        raise HTTPException(status_code=400, detail=f"Unknown variation axes: {', '.join(unknown)}")
    if not any(a.vary for a in req.axes):
        raise HTTPException(status_code=400, detail="At least one aspect must vary")

    family = _require_family(req.family_id)
    selected_characters = _selected_characters(req.character_ids)
    references = [p.strip() for p in req.reference_prompts if p.strip()]
    if not (req.user_input.strip() or req.aesthetic.strip() or selected_characters or references):
        raise HTTPException(
            status_code=400,
            detail="Give reference prompts, a base idea, an aesthetic or at least one character to keep constant",
        )

    axes = [a.model_dump() for a in req.axes]
    system = prompts.build_system_prompt("variations", family)
    user_message = prompts.build_user_message(
        "variations", req.user_input,
        example_prompts=req.example_prompts,
        characters=selected_characters,
        count=req.count,
        aesthetic=req.aesthetic,
        axes=axes,
        references=references,
    )

    logger.info(
        "variations request family_id=%s model=%s temperature=%.2f characters=%d references=%d "
        "count=%d varying=%s",
        req.family_id, req.llm_model, req.temperature, len(selected_characters), len(references),
        req.count, ",".join(a.key for a in req.axes if a.vary),
    )
    results = _call_llm("variations", req.llm_model, system, [user_message], req.temperature)
    pairs = prompts.parse_multi_response(results[0].content, family["has_negative_prompt"])

    return _record(
        "variations", family, results, pairs,
        expected_count=req.count,
        temperature=req.temperature,
        user_input=req.user_input,
        example_prompts=req.example_prompts,
        llm_model=req.llm_model,
        params={
            "count": req.count,
            "aesthetic": req.aesthetic,
            # Labels are stored so history stays readable without the axes table.
            "axes": [{**a, "label": prompts.VARIATION_AXES[a["key"]]} for a in axes],
            **({"reference_prompts": references} if references else {}),
            **_character_params(selected_characters),
        },
    )


@router.post("/from-image")
async def from_image(
    image: UploadFile = File(...),
    family_id: str = Form(...),
    vision_model: str = Form(...),
    user_input: str = Form(""),
    example_prompts: str = Form(""),
    temperature: float = Form(0.7),
    variants: int = Form(1, ge=1, le=MAX_VARIANTS),
):
    image_data_uri, image_bytes = await _read_image_data_uri(image)
    family = _require_family(family_id)

    system = prompts.build_system_prompt("image", family)
    user_messages = [
        prompts.build_user_message(
            "image", user_input, example_prompts=example_prompts, variant=(index, variants)
        )
        for index in range(1, variants + 1)
    ]

    logger.info(
        "from-image request family_id=%s vision_model=%s temperature=%.2f image_bytes=%d variants=%d",
        family_id, vision_model, temperature, image_bytes, variants,
    )
    results = _call_llm(
        "image", vision_model, system, user_messages, temperature, image_data_uri=image_data_uri
    )
    pairs = [prompts.parse_response(r.content, family["has_negative_prompt"]) for r in results]

    return _record(
        "image", family, results, pairs,
        expected_count=variants,
        temperature=temperature,
        user_input=user_input,
        example_prompts=example_prompts,
        vision_model=vision_model,
        params={"variants": variants},
    )


def _character_response(result: OpenRouterResult) -> dict:
    name, text = prompts.parse_character_response(result.content)
    logger.info(
        "extract-character request succeeded name_len=%d text_len=%d cost=%s",
        len(name), len(text), result.cost,
    )
    return {"name": name, "text": text, "cost": result.cost, "truncated": result.truncated}


class ExtractCharacterRequest(BaseModel):
    prompt_text: str
    llm_model: str
    temperature: float = 0.7


@router.post("/extract-character")
def extract_character(req: ExtractCharacterRequest):
    if not req.prompt_text.strip():
        raise HTTPException(status_code=400, detail="prompt_text is required")

    system = system_prompt.read_system_prompt("extract_character")
    user_message = f"## Prompt\n{req.prompt_text.strip()}"

    logger.info("extract-character request model=%s", req.llm_model)
    results = _call_llm(
        "extract-character", req.llm_model, system, [user_message], req.temperature
    )
    return _character_response(results[0])


@router.post("/extract-character-from-image")
async def extract_character_from_image(
    image: UploadFile = File(...),
    vision_model: str = Form(...),
    temperature: float = Form(0.7),
):
    image_data_uri, image_bytes = await _read_image_data_uri(image)

    system = system_prompt.read_system_prompt("extract_character")
    user_message = "## Image\nExtract the description of the main character in this image."

    logger.info(
        "extract-character-from-image request vision_model=%s image_bytes=%d",
        vision_model, image_bytes,
    )
    results = _call_llm(
        "extract-character-from-image", vision_model, system, [user_message], temperature,
        image_data_uri=image_data_uri,
    )
    return _character_response(results[0])


@router.get("/openrouter-models")
def get_openrouter_models():
    try:
        return list_models()
    except RuntimeError as e:
        raise HTTPException(status_code=502, detail=str(e))


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


@router.get("/characters")
def get_characters():
    return characters.list_characters()

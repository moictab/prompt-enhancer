# Per-Mode System Prompts + Generar/Iterar Merge Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the single shared global system prompt with three independent, admin-editable prompts (`generate`/`iterate`/`image`), and merge the `/api/generate` + `/api/iterate` endpoints and their UI tabs into one, so a single "existing prompt" field (empty = generate from scratch, filled = iterate) decides the mode.

**Architecture:** `data/system_prompts.json` (one JSON object, three keys) replaces `data/system_prompt.txt`. `app/prompts.py`'s `build_system_prompt` takes a `mode` string directly and internally fetches the matching prompt — no more `is_iteration` boolean or hardcoded `ITERATION_ADDENDUM`. `POST /api/generate` becomes the single entry point for both generate and iterate (an optional `previous_prompt` field decides which); `POST /api/iterate` is deleted. The frontend collapses the Generar and Iterar tabs into one.

**Tech Stack:** Same as the existing app — Python 3.10+, FastAPI, Jinja2, vanilla JS, `pytest`.

## Global Constraints

- Three independent global system prompts (`generate`, `iterate`, `image`), no shared base text between them — stored as one JSON object `data/system_prompts.json` with those three keys — see spec "Data Model".
- `ITERATION_ADDENDUM` is removed as a constant from `prompts.py`; its content now lives entirely in `system_prompts.json["iterate"]`, admin-editable — see spec "Data Model" / "Backend".
- `POST /api/iterate` is deleted. `POST /api/generate` handles both cases via an optional `previous_prompt` field: blank means generate-from-scratch, non-blank means iterate — see spec "Backend".
- `history.jsonl`'s shape is unchanged — `mode` still logs exactly `"generate"`, `"iterate"`, or `"image"` — see spec "Non-Goals".
- No migration of the old `data/system_prompt.txt` content — no prompt has been customized via the admin panel on any deployment yet, so the old file is simply superseded and left orphaned on disk — see spec "Non-Goals".
- Frontend goes from 3 tabs to 2 (Generar merged, Imagen); the "Iterar este prompt" button fills a field in the (now single) Generar tab rather than switching to a separate tab — see spec "Frontend".

---

## Task 1: `system_prompt.py` — Mode-Keyed Storage

**Files:**
- Modify: `app/system_prompt.py`
- Modify: `tests/test_system_prompt.py`

**Interfaces:**
- Consumes: `app.storage.read_json`, `app.storage.write_json` (existing), `app.config.get_settings` (existing).
- Produces (signature change): `app.system_prompt.read_system_prompt(mode: str, path: str | None = None) -> str`, `app.system_prompt.write_system_prompt(mode: str, text: str, path: str | None = None) -> None`. Both operate on a JSON object at `data/system_prompts.json` (default path), keyed by `mode`.

- [ ] **Step 1: Write the failing tests**

Replace `tests/test_system_prompt.py` in full:
```python
from app import system_prompt


def test_read_system_prompt_returns_empty_string_when_missing(tmp_path):
    path = str(tmp_path / "system_prompts.json")

    assert system_prompt.read_system_prompt("generate", path=path) == ""


def test_write_then_read_roundtrips(tmp_path):
    path = str(tmp_path / "system_prompts.json")

    system_prompt.write_system_prompt("generate", "You are an expert prompt engineer.", path=path)

    assert system_prompt.read_system_prompt("generate", path=path) == "You are an expert prompt engineer."


def test_write_creates_parent_directories(tmp_path):
    path = str(tmp_path / "nested" / "system_prompts.json")

    system_prompt.write_system_prompt("generate", "hello", path=path)

    assert (tmp_path / "nested" / "system_prompts.json").exists()


def test_modes_are_stored_independently(tmp_path):
    path = str(tmp_path / "system_prompts.json")

    system_prompt.write_system_prompt("generate", "generate text", path=path)
    system_prompt.write_system_prompt("iterate", "iterate text", path=path)
    system_prompt.write_system_prompt("image", "image text", path=path)

    assert system_prompt.read_system_prompt("generate", path=path) == "generate text"
    assert system_prompt.read_system_prompt("iterate", path=path) == "iterate text"
    assert system_prompt.read_system_prompt("image", path=path) == "image text"


def test_writing_one_mode_does_not_clobber_another(tmp_path):
    path = str(tmp_path / "system_prompts.json")
    system_prompt.write_system_prompt("generate", "generate text", path=path)

    system_prompt.write_system_prompt("iterate", "iterate text", path=path)

    assert system_prompt.read_system_prompt("generate", path=path) == "generate text"
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
pytest tests/test_system_prompt.py -v
```
Expected: FAIL (`TypeError: read_system_prompt() missing 1 required positional argument: 'mode'` or similar, since the current signature takes no `mode`).

- [ ] **Step 3: Rewrite `app/system_prompt.py`**

```python
from . import storage
from .config import get_settings


def _default_path() -> str:
    return f"{get_settings().data_dir}/system_prompts.json"


def read_system_prompt(mode: str, path: str | None = None) -> str:
    data = storage.read_json(path or _default_path(), default={})
    return data.get(mode, "")


def write_system_prompt(mode: str, text: str, path: str | None = None) -> None:
    resolved_path = path or _default_path()
    data = storage.read_json(resolved_path, default={})
    data[mode] = text
    storage.write_json(resolved_path, data)
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
pytest tests/test_system_prompt.py -v
```
Expected: PASS (all 5 tests).

- [ ] **Step 5: Commit**

```bash
git add app/system_prompt.py tests/test_system_prompt.py
git commit -m "Store system prompts as a mode-keyed JSON object instead of a single file"
```

---

## Task 2: Seed Data — Three Default Prompts

**Files:**
- Modify: `app/seed.py`
- Modify: `tests/test_seed.py`

**Interfaces:**
- Consumes: `app.system_prompt.write_system_prompt(mode, text, path=None)` (Task 1).
- Produces: `ensure_seed_data` now seeds `data/system_prompts.json` with three keys (`generate`, `iterate`, `image`) instead of writing `data/system_prompt.txt`.

- [ ] **Step 1: Write the failing tests**

Replace `tests/test_seed.py` in full:
```python
import json

from app.seed import ensure_seed_data


def test_ensure_seed_data_creates_families_with_sdxl_and_zit(tmp_path):
    ensure_seed_data(str(tmp_path))

    families = json.loads((tmp_path / "families.json").read_text())
    names = {f["name"] for f in families}
    assert names == {"SDXL", "Z-Image-Turbo"}

    sdxl = next(f for f in families if f["name"] == "SDXL")
    zit = next(f for f in families if f["name"] == "Z-Image-Turbo")
    assert sdxl["has_negative_prompt"] is True
    assert zit["has_negative_prompt"] is False


def test_ensure_seed_data_creates_empty_characters_file(tmp_path):
    ensure_seed_data(str(tmp_path))

    characters = json.loads((tmp_path / "characters.json").read_text())
    assert characters == []


def test_ensure_seed_data_creates_all_three_system_prompts(tmp_path):
    ensure_seed_data(str(tmp_path))

    data = json.loads((tmp_path / "system_prompts.json").read_text())
    assert set(data.keys()) == {"generate", "iterate", "image"}
    assert "POSITIVE:" in data["generate"]
    assert "POSITIVE:" in data["iterate"]
    assert "POSITIVE:" in data["image"]
    assert "PRESERVE" in data["iterate"]


def test_ensure_seed_data_is_idempotent(tmp_path):
    ensure_seed_data(str(tmp_path))
    ensure_seed_data(str(tmp_path))

    families = json.loads((tmp_path / "families.json").read_text())
    assert len(families) == 2


def test_ensure_seed_data_does_not_overwrite_existing_system_prompts(tmp_path):
    (tmp_path / "system_prompts.json").write_text('{"generate": "custom"}', encoding="utf-8")

    ensure_seed_data(str(tmp_path))

    data = json.loads((tmp_path / "system_prompts.json").read_text())
    assert data == {"generate": "custom"}
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
pytest tests/test_seed.py -v
```
Expected: FAIL (`test_ensure_seed_data_creates_all_three_system_prompts` and the overwrite test fail — `system_prompts.json` doesn't exist yet; the old `system_prompt.txt`-based behavior is still in place).

- [ ] **Step 3: Rewrite `app/seed.py`**

```python
from pathlib import Path

from . import families, storage, system_prompt

DEFAULT_GENERATE_SYSTEM_PROMPT = """You are an expert AI image-generation prompt engineer. The user will describe a new image idea in natural language -- however detailed or vague. Transform it into a high-quality prompt for the selected model family, following that family's specific rules exactly. Fill in reasonable creative details where the user was vague, without contradicting anything they specified.

Be specific and vivid. Replace vague words with concrete descriptions.

You MUST output in exactly this format:
POSITIVE: <the enhanced positive prompt>
NEGATIVE: <the negative prompt, or leave empty if the family doesn't use one>
"""

DEFAULT_ITERATE_SYSTEM_PROMPT = """You are an expert AI image-generation prompt engineer. The user is refining an EXISTING prompt. You will receive the previous prompt that was already generated and the user's requested changes.

Your job is to:
- PRESERVE the good elements from the previous prompt -- do not regenerate from scratch
- Apply the user's requested changes precisely
- Maintain the same overall style and structure
- Only modify what the user explicitly asks to change
- If the user asks to "add" something, integrate it naturally into the existing prompt
- If the user asks to "remove" something, take it out cleanly without leaving gaps

Follow the selected model family's specific rules exactly.

You MUST output in exactly this format:
POSITIVE: <the enhanced positive prompt>
NEGATIVE: <the negative prompt, or leave empty if the family doesn't use one>
"""

DEFAULT_IMAGE_SYSTEM_PROMPT = """You are an expert AI image-generation prompt engineer. The user has attached a reference image, optionally with additional text describing further intent. Analyze the image's subject, composition, lighting, and style, and translate what you observe into a prompt following the selected model family's rules. If additional text is provided, prioritize it for adjustments (e.g., style changes, additions) while keeping the image as the primary source of truth for content.

Be specific and vivid. Replace vague words with concrete descriptions.

You MUST output in exactly this format:
POSITIVE: <the enhanced positive prompt>
NEGATIVE: <the negative prompt, or leave empty if the family doesn't use one>
"""

SDXL_INSTRUCTIONS = """## Prompt Structure
Always follow this order: Subject -> Environment -> Lighting -> Details -> Style

## Rules
- Write in natural language with light emphasis weighting using (word:1.1) to (word:1.4) syntax. NEVER exceed 1.5.
- Do NOT spam quality tags like "masterpiece, best quality, 4k, 8k, ultra detailed". These dilute the actual prompt content.
- Aim for 40-70 tokens. SDXL has a 77-token CLIP limit per chunk -- stay concise and impactful.
- Use photographic terms when appropriate: camera model, lens type, focal length, aperture, film stock.
- Include artistic style references (artist names, art movements, media types) when they serve the concept.

## Negative Prompt
Generate a focused negative prompt of 15-30 tokens targeting real artifacts to avoid (e.g., blurry, deformed hands, extra fingers, watermark, text). Do NOT pad with generic quality negatives.
"""

ZIT_INSTRUCTIONS = """## Critical Differences from Other Models
Z-Image-Turbo uses a COMPLETELY different architecture. You must follow these rules strictly:

## Prompt Structure (4-6 Layers)
Follow this strict hierarchy:
1. Subject & Action -- who/what is doing what
2. Environment -- where, surrounding context
3. Style -- artistic style, medium, aesthetic
4. Lighting -- light sources, quality, direction
5. Composition -- camera angle, framing, depth
6. Constraints -- what to exclude (anti-hallucination)

## Rules
- Use PURE natural language ONLY. No weight syntax, no tags, no brackets, no parentheses for emphasis.
- NEVER use quality tags ("masterpiece", "best quality", "4k", "8k"). These actively cause hallucination artifacts in ZIT.
- Emphasize prepositions -- they control spatial relationships in ZIT. Words like "beneath", "towering above", "nestled between", "reflected in" are powerful.
- Target 80-250 words. Sweet spot is 120-180 words. Focus on 3-5 key concepts maximum.
- End every prompt with anti-hallucination language: "no text, no watermarks, no logos, no signatures, no borders, no frames"
- Write as flowing, descriptive prose -- like a detailed scene description in a novel.

## Negative Prompt
Do NOT generate a negative prompt. Z-Image-Turbo uses guidance_scale=0.0, so negative prompts are completely ignored.
"""


def ensure_seed_data(data_dir: str) -> None:
    families_path = f"{data_dir}/families.json"
    characters_path = f"{data_dir}/characters.json"
    system_prompts_path = f"{data_dir}/system_prompts.json"

    if not Path(families_path).exists():
        families.create_family("SDXL", SDXL_INSTRUCTIONS, True, path=families_path)
        families.create_family("Z-Image-Turbo", ZIT_INSTRUCTIONS, False, path=families_path)

    if not Path(characters_path).exists():
        storage.write_json(characters_path, [])

    if not Path(system_prompts_path).exists():
        system_prompt.write_system_prompt("generate", DEFAULT_GENERATE_SYSTEM_PROMPT, path=system_prompts_path)
        system_prompt.write_system_prompt("iterate", DEFAULT_ITERATE_SYSTEM_PROMPT, path=system_prompts_path)
        system_prompt.write_system_prompt("image", DEFAULT_IMAGE_SYSTEM_PROMPT, path=system_prompts_path)
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
pytest tests/test_seed.py -v
```
Expected: PASS (all 5 tests).

- [ ] **Step 5: Run the full suite**

```bash
pytest -v
```
Expected: PASS (Tasks 1-2 tests all green; nothing else touched yet).

- [ ] **Step 6: Commit**

```bash
git add app/seed.py tests/test_seed.py
git commit -m "Seed three independent default system prompts (generate/iterate/image)"
```

---

## Task 3: `prompts.py` — Mode-Driven System Prompt Assembly

**Files:**
- Modify: `app/prompts.py`
- Modify: `tests/test_prompts.py`

**Interfaces:**
- Consumes: `app.system_prompt.read_system_prompt(mode, path=None)` (Task 1).
- Produces (signature change): `app.prompts.build_system_prompt(mode: str, family: dict) -> str` — drops the old `global_system_prompt`/`is_iteration` parameters, takes `mode` directly and internally calls `system_prompt.read_system_prompt(mode)`. `build_user_message` and `parse_response` are unchanged.

- [ ] **Step 1: Write the failing tests**

In `tests/test_prompts.py`, replace the two `build_system_prompt` tests (leave `build_user_message`/`parse_response` tests untouched):
```python
from unittest.mock import patch

from app.prompts import build_system_prompt, build_user_message, parse_response


@patch("app.prompts.system_prompt.read_system_prompt")
def test_build_system_prompt_combines_mode_prompt_and_family(mock_read):
    mock_read.return_value = "global rules"
    family = {"instructions": "family rules"}

    result = build_system_prompt("generate", family)

    assert result == "global rules\n\nfamily rules"
    mock_read.assert_called_once_with("generate")


@patch("app.prompts.system_prompt.read_system_prompt")
def test_build_system_prompt_uses_iterate_mode_prompt(mock_read):
    mock_read.return_value = "iterate rules"
    family = {"instructions": "family rules"}

    result = build_system_prompt("iterate", family)

    assert result == "iterate rules\n\nfamily rules"
    mock_read.assert_called_once_with("iterate")
```
(The rest of the file — `test_build_user_message_*` and `test_parse_response_*` — stays exactly as-is.)

- [ ] **Step 2: Run the tests to verify they fail**

```bash
pytest tests/test_prompts.py -v
```
Expected: FAIL (`TypeError: build_system_prompt() missing 1 required positional argument` or similar — current signature is `(global_system_prompt, family, is_iteration=False)`).

- [ ] **Step 3: Rewrite `app/prompts.py`**

```python
import re

from . import system_prompt


def build_system_prompt(mode: str, family: dict) -> str:
    return f"{system_prompt.read_system_prompt(mode).strip()}\n\n{family['instructions'].strip()}"


def build_user_message(
    mode: str,
    user_input: str,
    previous_prompt: str | None = None,
    example_prompts: str | None = None,
) -> str:
    parts = []
    if mode == "iterate":
        parts.append(f"## Prompt Previo\n{previous_prompt}")
        parts.append(f"## Cambios Solicitados\n{user_input}")
    elif mode == "image":
        parts.append("## Imagen\nSe adjunta una imagen de referencia para este prompt.")
        if user_input and user_input.strip():
            parts.append(f"## Idea\n{user_input.strip()}")
    else:
        parts.append(f"## Idea\n{user_input}")

    if example_prompts and example_prompts.strip():
        parts.append(f"## Prompts de Ejemplo\n{example_prompts.strip()}")

    return "\n\n".join(parts)


def parse_response(response: str, has_negative_prompt: bool) -> tuple[str, str]:
    positive = ""
    negative = ""

    response_upper = response.upper()
    pos_idx = response_upper.find("POSITIVE:")
    neg_idx = response_upper.find("NEGATIVE:")

    if pos_idx != -1:
        pos_start = pos_idx + len("POSITIVE:")
        if neg_idx != -1 and neg_idx > pos_idx:
            positive = response[pos_start:neg_idx].strip()
        else:
            positive = response[pos_start:].strip()
        if neg_idx != -1:
            neg_start = neg_idx + len("NEGATIVE:")
            negative = response[neg_start:].strip()
    else:
        lines = response.strip().split("\n")
        preamble_pattern = re.compile(
            r"^(sure|here|i |i'|of course|certainly|let me|this is|note:|hope|feel free)",
            re.IGNORECASE,
        )
        while lines and preamble_pattern.match(lines[0].strip()):
            lines.pop(0)
        while lines and preamble_pattern.match(lines[-1].strip()):
            lines.pop()
        positive = "\n".join(lines).strip()

    if not has_negative_prompt:
        negative = ""

    return (positive, negative)
```

Note: `ITERATION_ADDENDUM` is gone entirely — its content now lives in `data/system_prompts.json["iterate"]` (Task 2's seed data), not in this file.

- [ ] **Step 4: Run the tests to verify they pass**

```bash
pytest tests/test_prompts.py -v
```
Expected: PASS (all tests, including the untouched `build_user_message`/`parse_response` ones).

- [ ] **Step 5: Run the full suite**

```bash
pytest -v
```
Expected: FAIL for `tests/test_api_generate.py`, `tests/test_api_iterate.py`, and `tests/test_api_from_image.py` (they still call the old `build_system_prompt` signature indirectly through `app/routes/api.py`, which Task 4 fixes). This is expected at this point in the plan — do not attempt to fix `routes/api.py` in this task.

- [ ] **Step 6: Commit**

```bash
git add app/prompts.py tests/test_prompts.py
git commit -m "Make build_system_prompt mode-driven, remove ITERATION_ADDENDUM constant"
```

---

## Task 4: Merge `/api/generate` and `/api/iterate`

**Files:**
- Modify: `app/routes/api.py`
- Modify: `tests/test_api_generate.py`
- Delete: `tests/test_api_iterate.py`

**Interfaces:**
- Consumes: `app.prompts.build_system_prompt(mode, family)` (Task 3), `app.prompts.build_user_message(mode, ...)` (unchanged), `app.families.get_family`, `app.history.append_entry`, `app.openrouter_client.call_openrouter` (all unchanged).
- Produces: `GenerateRequest` gains `previous_prompt: str = ""`. `POST /api/iterate` and `IterateRequest` are deleted. `POST /api/generate` decides `mode` internally from whether `previous_prompt` is blank.

- [ ] **Step 1: Write the failing tests**

Replace `tests/test_api_generate.py` in full:
```python
from unittest.mock import patch

from app import families


def _create_family(tmp_path, has_negative_prompt=True):
    return families.create_family(
        "SDXL", "family instructions", has_negative_prompt,
        path=str(tmp_path / "families.json"),
    )


def test_generate_requires_auth(api_client):
    response = api_client.post("/api/generate", json={
        "user_input": "a cat", "family_id": "x", "llm_model": "m",
    })

    assert response.status_code == 401


@patch("app.routes.api.call_openrouter")
def test_generate_returns_parsed_prompt(mock_call, api_client, auth_headers, tmp_path):
    family = _create_family(tmp_path)
    mock_call.return_value = "POSITIVE: a cyberpunk samurai\nNEGATIVE: blurry"

    response = api_client.post(
        "/api/generate",
        json={
            "user_input": "a cyberpunk samurai",
            "family_id": family["id"],
            "llm_model": "anthropic/claude-sonnet-4",
            "temperature": 0.7,
        },
        auth=auth_headers,
    )

    assert response.status_code == 200
    assert response.json() == {
        "positive_prompt": "a cyberpunk samurai",
        "negative_prompt": "blurry",
    }


@patch("app.routes.api.call_openrouter")
def test_generate_appends_to_history_as_generate_mode(mock_call, api_client, auth_headers, tmp_path):
    family = _create_family(tmp_path)
    mock_call.return_value = "POSITIVE: a cat\nNEGATIVE: blurry"

    api_client.post(
        "/api/generate",
        json={"user_input": "a cat", "family_id": family["id"], "llm_model": "m"},
        auth=auth_headers,
    )

    from app import history
    entries = history.list_entries(path=str(tmp_path / "history.jsonl"))
    assert len(entries) == 1
    assert entries[0]["mode"] == "generate"
    assert entries[0]["family_name"] == "SDXL"
    assert entries[0]["previous_prompt"] is None


def test_generate_rejects_blank_user_input(api_client, auth_headers, tmp_path):
    family = _create_family(tmp_path)

    response = api_client.post(
        "/api/generate",
        json={"user_input": "   ", "family_id": family["id"], "llm_model": "m"},
        auth=auth_headers,
    )

    assert response.status_code == 400


def test_generate_rejects_unknown_family(api_client, auth_headers):
    response = api_client.post(
        "/api/generate",
        json={"user_input": "a cat", "family_id": "nonexistent", "llm_model": "m"},
        auth=auth_headers,
    )

    assert response.status_code == 404


@patch("app.routes.api.call_openrouter")
def test_generate_returns_502_on_openrouter_error(mock_call, api_client, auth_headers, tmp_path):
    family = _create_family(tmp_path)
    mock_call.side_effect = RuntimeError("OpenRouter rate limit exceeded. Wait a moment and try again.")

    response = api_client.post(
        "/api/generate",
        json={"user_input": "a cat", "family_id": family["id"], "llm_model": "m"},
        auth=auth_headers,
    )

    assert response.status_code == 502
    assert "rate limit" in response.json()["detail"]


@patch("app.routes.api.call_openrouter")
def test_generate_with_previous_prompt_returns_parsed_prompt(mock_call, api_client, auth_headers, tmp_path):
    family = _create_family(tmp_path)
    mock_call.return_value = "POSITIVE: a samurai with lightning\nNEGATIVE: blurry"

    response = api_client.post(
        "/api/generate",
        json={
            "user_input": "add lightning",
            "previous_prompt": "a samurai in rain",
            "family_id": family["id"],
            "llm_model": "m",
        },
        auth=auth_headers,
    )

    assert response.status_code == 200
    assert response.json() == {
        "positive_prompt": "a samurai with lightning",
        "negative_prompt": "blurry",
    }


@patch("app.routes.api.call_openrouter")
def test_generate_with_previous_prompt_appends_to_history_as_iterate_mode(mock_call, api_client, auth_headers, tmp_path):
    family = _create_family(tmp_path)
    mock_call.return_value = "POSITIVE: updated\nNEGATIVE: "

    api_client.post(
        "/api/generate",
        json={
            "user_input": "add lightning", "previous_prompt": "a samurai",
            "family_id": family["id"], "llm_model": "m",
        },
        auth=auth_headers,
    )

    from app import history
    entries = history.list_entries(path=str(tmp_path / "history.jsonl"))
    assert entries[0]["mode"] == "iterate"
    assert entries[0]["previous_prompt"] == "a samurai"


@patch("app.routes.api.call_openrouter")
def test_generate_treats_blank_previous_prompt_as_generate_mode(mock_call, api_client, auth_headers, tmp_path):
    family = _create_family(tmp_path)
    mock_call.return_value = "POSITIVE: a cat\nNEGATIVE: "

    api_client.post(
        "/api/generate",
        json={
            "user_input": "a cat", "previous_prompt": "   ",
            "family_id": family["id"], "llm_model": "m",
        },
        auth=auth_headers,
    )

    from app import history
    entries = history.list_entries(path=str(tmp_path / "history.jsonl"))
    assert entries[0]["mode"] == "generate"
    assert entries[0]["previous_prompt"] is None
```

Delete `tests/test_api_iterate.py`:
```bash
git rm tests/test_api_iterate.py
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
pytest tests/test_api_generate.py -v
```
Expected: FAIL — `test_generate_with_previous_prompt_*` and `test_generate_treats_blank_previous_prompt_as_generate_mode` fail with a `422` (unrecognized `previous_prompt` field is currently silently accepted by Pydantic and ignored, or fails validation depending on config) since `GenerateRequest` doesn't have that field yet, and `/api/iterate` still exists as a separate route.

- [ ] **Step 3: Rewrite `app/routes/api.py`**

```python
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

    settings = get_settings()
    system = prompts.build_system_prompt(mode, family)
    user_message = prompts.build_user_message(
        mode, req.user_input,
        previous_prompt=req.previous_prompt or None,
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
        previous_prompt=req.previous_prompt or None,
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
```

Note: `system_prompt` is no longer imported in this file — `prompts.build_system_prompt` now fetches the right prompt internally (Task 3).

- [ ] **Step 4: Run the tests to verify they pass**

```bash
pytest tests/test_api_generate.py -v
```
Expected: PASS (all 9 tests).

- [ ] **Step 5: Run the full suite**

```bash
pytest -v
```
Expected: FAIL only for `tests/test_api_from_image.py` if it still references the old `build_system_prompt` call shape indirectly — check the failure; if `from_image`'s call to `prompts.build_system_prompt("image", family)` matches what Task 3 produced, `test_api_from_image.py` should already pass unchanged (it never asserted on `build_system_prompt`'s internals, only on the final parsed response). If it fails, read the failure output before assuming — do not silently change `test_api_from_image.py` without understanding why.

- [ ] **Step 6: Commit**

```bash
git add app/routes/api.py tests/test_api_generate.py
git rm tests/test_api_iterate.py
git commit -m "Merge /api/iterate into /api/generate via optional previous_prompt"
```

---

## Task 5: Admin Routes — Per-Mode System Prompt Endpoints

**Files:**
- Modify: `app/routes/admin.py`
- Modify: `tests/test_admin.py`

**Interfaces:**
- Consumes: `app.system_prompt.read_system_prompt(mode, path=None)` / `write_system_prompt(mode, text, path=None)` (Task 1).
- Produces: `GET`/`PUT /api/admin/system-prompt` become `GET`/`PUT /api/admin/system-prompt/{mode}`, validating `mode` is one of `generate`/`iterate`/`image` (400 otherwise).

- [ ] **Step 1: Write the failing tests**

In `tests/test_admin.py`, replace `test_get_and_update_system_prompt` with:
```python
def test_get_and_update_system_prompt_for_each_mode(api_client, auth_headers):
    for mode in ["generate", "iterate", "image"]:
        get_response = api_client.get(f"/api/admin/system-prompt/{mode}", auth=auth_headers)
        assert get_response.status_code == 200
        assert get_response.json() == {"text": ""}

        put_response = api_client.put(
            f"/api/admin/system-prompt/{mode}",
            json={"text": f"You are an expert at {mode}."},
            auth=auth_headers,
        )
        assert put_response.status_code == 200

        assert api_client.get(f"/api/admin/system-prompt/{mode}", auth=auth_headers).json() == {
            "text": f"You are an expert at {mode}."
        }


def test_system_prompt_modes_are_independent(api_client, auth_headers):
    api_client.put(
        "/api/admin/system-prompt/generate", json={"text": "generate text"}, auth=auth_headers
    )

    assert api_client.get("/api/admin/system-prompt/iterate", auth=auth_headers).json() == {
        "text": ""
    }


def test_get_system_prompt_rejects_unknown_mode(api_client, auth_headers):
    response = api_client.get("/api/admin/system-prompt/bogus", auth=auth_headers)

    assert response.status_code == 400


def test_update_system_prompt_rejects_unknown_mode(api_client, auth_headers):
    response = api_client.put(
        "/api/admin/system-prompt/bogus", json={"text": "x"}, auth=auth_headers
    )

    assert response.status_code == 400
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
pytest tests/test_admin.py -v
```
Expected: FAIL (`404 Not Found` for `/api/admin/system-prompt/generate` etc. — the route is currently `/api/admin/system-prompt` with no `{mode}` path segment).

- [ ] **Step 3: Update `app/routes/admin.py`**

Replace the system-prompt section at the bottom of the file (everything from `class SystemPromptPayload` onward):
```python
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
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
pytest tests/test_admin.py -v
```
Expected: PASS (all tests, including the 4 new ones).

- [ ] **Step 5: Run the full suite**

```bash
pytest -v
```
Expected: PASS for everything except `tests/test_pages.py`'s admin/index markup assertions (Task 6/7 fix those).

- [ ] **Step 6: Commit**

```bash
git add app/routes/admin.py tests/test_admin.py
git commit -m "Parameterize admin system-prompt routes by mode"
```

---

## Task 6: Frontend — Merge Generar and Iterar Tabs

**Files:**
- Modify: `app/templates/index.html`
- Modify: `app/static/app.js`
- Modify: `tests/test_pages.py`

**Interfaces:**
- Consumes: `POST /api/generate` (now handling both modes, Task 4).
- Produces: two tabs instead of three (`Generar`, `Imagen`); the Generar form gains a `generar-previous-prompt` field; `setupIterarForm` and the standalone Iterar section are removed.

- [ ] **Step 1: Replace `app/templates/index.html` in full**

```html
{% extends "base.html" %}
{% block title %}Prompt Enhancer{% endblock %}
{% block content %}
<div class="tabs">
  <button class="tab-button active" data-tab="generar" type="button">Generar</button>
  <button class="tab-button" data-tab="imagen" type="button">Imagen</button>
</div>

<section id="tab-generar" class="tab-panel active">
  <form id="form-generar">
    <label for="generar-previous-prompt">Prompt existente (dejalo vacio para generar desde cero)</label>
    <textarea id="generar-previous-prompt" name="previous_prompt" rows="4"></textarea>

    <label for="generar-user-input">Idea / cambios solicitados</label>
    <textarea id="generar-user-input" name="user_input" rows="4" required></textarea>

    <div class="characters" id="generar-characters"></div>

    <label for="generar-family">Familia de modelo</label>
    <select id="generar-family" name="family_id" required>
      {% for family in families %}
      <option value="{{ family.id }}">{{ family.name }}</option>
      {% endfor %}
    </select>

    <label for="generar-example-prompts">Prompts de ejemplo (opcional)</label>
    <textarea id="generar-example-prompts" name="example_prompts" rows="3"></textarea>

    <label for="generar-llm-model">Modelo LLM (OpenRouter)</label>
    <input id="generar-llm-model" name="llm_model" type="text" value="anthropic/claude-sonnet-4" required>

    <label for="generar-creativity">Creatividad: <span class="creativity-value">0.7</span></label>
    <input id="generar-creativity" name="temperature" type="range" min="0" max="1" step="0.05" value="0.7">

    <button type="submit">Generar</button>
  </form>
</section>

<section id="tab-imagen" class="tab-panel">
  <form id="form-imagen">
    <label for="imagen-file">Imagen</label>
    <input id="imagen-file" name="image" type="file" accept="image/jpeg,image/png,image/webp" required>
    <img id="imagen-preview" class="image-preview" hidden alt="Previsualizacion">

    <label for="imagen-user-input">Idea adicional (opcional)</label>
    <textarea id="imagen-user-input" name="user_input" rows="3"></textarea>

    <label for="imagen-family">Familia de modelo</label>
    <select id="imagen-family" name="family_id" required>
      {% for family in families %}
      <option value="{{ family.id }}">{{ family.name }}</option>
      {% endfor %}
    </select>

    <label for="imagen-example-prompts">Prompts de ejemplo (opcional)</label>
    <textarea id="imagen-example-prompts" name="example_prompts" rows="3"></textarea>

    <label for="imagen-vision-model">Modelo de vision (OpenRouter)</label>
    <input id="imagen-vision-model" name="vision_model" type="text" value="anthropic/claude-sonnet-4" required>

    <label for="imagen-creativity">Creatividad: <span class="creativity-value">0.7</span></label>
    <input id="imagen-creativity" name="temperature" type="range" min="0" max="1" step="0.05" value="0.7">

    <button type="submit">Generar desde imagen</button>
  </form>
</section>

<section id="result-panel" class="result-panel" hidden>
  <div id="result-error" class="error-banner" hidden></div>
  <div id="result-success" hidden>
    <label>Positive Prompt</label>
    <div class="copy-row">
      <textarea id="result-positive" readonly rows="4"></textarea>
      <button type="button" id="copy-positive">Copiar</button>
    </div>
    <div id="result-negative-group">
      <label>Negative Prompt</label>
      <div class="copy-row">
        <textarea id="result-negative" readonly rows="2"></textarea>
        <button type="button" id="copy-negative">Copiar</button>
      </div>
    </div>
    <button type="button" id="iterate-this">Iterar este prompt</button>
  </div>
</section>
{% endblock %}
{% block scripts %}
<script src="/static/app.js"></script>
{% endblock %}
```

- [ ] **Step 2: Replace `app/static/app.js` in full**

```javascript
function setupTabs() {
  const buttons = document.querySelectorAll(".tab-button");
  buttons.forEach((button) => {
    button.addEventListener("click", () => {
      buttons.forEach((b) => b.classList.remove("active"));
      document.querySelectorAll(".tab-panel").forEach((p) => p.classList.remove("active"));
      button.classList.add("active");
      document.getElementById(`tab-${button.dataset.tab}`).classList.add("active");
    });
  });
}

function setupCreativitySliders() {
  document.querySelectorAll('input[type="range"]').forEach((slider) => {
    const label = slider.closest("form").querySelector(".creativity-value");
    slider.addEventListener("input", () => {
      label.textContent = slider.value;
    });
  });
}

function showError(message) {
  document.getElementById("result-panel").hidden = false;
  const errorBox = document.getElementById("result-error");
  errorBox.hidden = false;
  errorBox.textContent = message;
  document.getElementById("result-success").hidden = true;
}

function showResult(positive, negative) {
  document.getElementById("result-panel").hidden = false;
  document.getElementById("result-error").hidden = true;
  document.getElementById("result-success").hidden = false;
  document.getElementById("result-positive").value = positive;
  document.getElementById("result-negative").value = negative;
  document.getElementById("result-negative-group").hidden = !negative;
}

async function postJSON(url, payload) {
  const response = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  const data = await response.json();
  if (!response.ok) {
    throw new Error(data.detail || "Error desconocido");
  }
  return data;
}

function setupGenerarForm() {
  const form = document.getElementById("form-generar");
  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    const formData = new FormData(form);
    try {
      const data = await postJSON("/api/generate", {
        user_input: formData.get("user_input"),
        previous_prompt: formData.get("previous_prompt"),
        family_id: formData.get("family_id"),
        example_prompts: formData.get("example_prompts"),
        llm_model: formData.get("llm_model"),
        temperature: parseFloat(formData.get("temperature")),
      });
      showResult(data.positive_prompt, data.negative_prompt);
    } catch (err) {
      showError(err.message);
    }
  });
}

function setupCopyButtons() {
  document.getElementById("copy-positive").addEventListener("click", () => {
    navigator.clipboard.writeText(document.getElementById("result-positive").value);
  });
  document.getElementById("copy-negative").addEventListener("click", () => {
    navigator.clipboard.writeText(document.getElementById("result-negative").value);
  });
}

function setupIterateHandoff() {
  document.getElementById("iterate-this").addEventListener("click", () => {
    const positive = document.getElementById("result-positive").value;
    document.getElementById("generar-previous-prompt").value = positive;
    document.querySelector('.tab-button[data-tab="generar"]').click();
  });
}

function insertAtCursor(textarea, text) {
  const start = textarea.selectionStart;
  const end = textarea.selectionEnd;
  const before = textarea.value.slice(0, start);
  const after = textarea.value.slice(end);
  textarea.value = `${before}${text}${after}`;
  const cursor = start + text.length;
  textarea.selectionStart = cursor;
  textarea.selectionEnd = cursor;
  textarea.focus();
}

async function loadCharacters() {
  const response = await fetch("/api/characters");
  if (!response.ok) return [];
  return response.json();
}

async function setupCharacterButtons() {
  const characterList = await loadCharacters();
  const container = document.getElementById("generar-characters");
  const textarea = document.getElementById("generar-user-input");
  characterList.forEach((character) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "character-button";
    button.textContent = character.name;
    button.addEventListener("click", () => insertAtCursor(textarea, character.text));
    container.appendChild(button);
  });
}

function setupImagenForm() {
  const fileInput = document.getElementById("imagen-file");
  const preview = document.getElementById("imagen-preview");
  fileInput.addEventListener("change", () => {
    const file = fileInput.files[0];
    if (!file) {
      preview.hidden = true;
      return;
    }
    preview.src = URL.createObjectURL(file);
    preview.hidden = false;
  });

  const form = document.getElementById("form-imagen");
  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    const formData = new FormData(form);
    try {
      const response = await fetch("/api/from-image", { method: "POST", body: formData });
      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.detail || "Error desconocido");
      }
      showResult(data.positive_prompt, data.negative_prompt);
    } catch (err) {
      showError(err.message);
    }
  });
}

document.addEventListener("DOMContentLoaded", () => {
  setupTabs();
  setupCreativitySliders();
  setupGenerarForm();
  setupImagenForm();
  setupCopyButtons();
  setupIterateHandoff();
  setupCharacterButtons();
});
```

- [ ] **Step 3: Update `tests/test_pages.py`**

Replace `test_index_renders_all_three_tabs` with:
```python
def test_index_renders_generar_and_imagen_tabs(api_client, auth_headers):
    response = api_client.get("/", auth=auth_headers)

    assert response.status_code == 200
    assert 'id="form-generar"' in response.text
    assert 'id="form-imagen"' in response.text
    assert 'id="generar-previous-prompt"' in response.text
    assert 'id="form-iterar"' not in response.text
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
pytest tests/test_pages.py -v
```
Expected: PASS for the index-page tests (the admin-page test still fails until Task 7).

- [ ] **Step 5: Manually smoke-test in the browser**

```bash
uvicorn app.main:app --reload
```
Open the app, confirm only 2 tabs appear (Generar, Imagen), confirm the "Prompt existente" field is present and empty by default, and confirm character buttons still insert into the idea field.

- [ ] **Step 6: Commit**

```bash
git add app/templates/index.html app/static/app.js tests/test_pages.py
git commit -m "Merge Generar and Iterar into a single tab"
```

---

## Task 7: Frontend — Three System Prompt Sections in Admin

**Files:**
- Modify: `app/templates/admin.html`
- Modify: `app/static/admin.js`
- Modify: `tests/test_pages.py`

**Interfaces:**
- Consumes: `GET`/`PUT /api/admin/system-prompt/{mode}` (Task 5).
- Produces: three independent textarea + save-button pairs (Generar / Iterar / Imagen) replacing the single "System Prompt Global" section.

- [ ] **Step 1: Update `app/templates/admin.html`**

Replace the single "System Prompt Global" `<section>` (the first one) with three:
```html
<section>
  <h2>System Prompt: Generar</h2>
  <textarea id="system-prompt-generate-text" rows="8"></textarea>
  <button type="button" id="save-system-prompt-generate">Guardar</button>
</section>

<section>
  <h2>System Prompt: Iterar</h2>
  <textarea id="system-prompt-iterate-text" rows="8"></textarea>
  <button type="button" id="save-system-prompt-iterate">Guardar</button>
</section>

<section>
  <h2>System Prompt: Imagen</h2>
  <textarea id="system-prompt-image-text" rows="8"></textarea>
  <button type="button" id="save-system-prompt-image">Guardar</button>
</section>
```
(The Familias and Personajes sections below it are untouched.)

- [ ] **Step 2: Update `app/static/admin.js`**

Replace `loadSystemPrompt` and `setupSystemPromptForm` with:
```javascript
const SYSTEM_PROMPT_MODES = ["generate", "iterate", "image"];

async function loadSystemPrompts() {
  for (const mode of SYSTEM_PROMPT_MODES) {
    const data = await fetchJSON(`/api/admin/system-prompt/${mode}`);
    document.getElementById(`system-prompt-${mode}-text`).value = data.text;
  }
}

function setupSystemPromptForms() {
  SYSTEM_PROMPT_MODES.forEach((mode) => {
    document.getElementById(`save-system-prompt-${mode}`).addEventListener("click", async () => {
      const text = document.getElementById(`system-prompt-${mode}-text`).value;
      await fetchJSON(`/api/admin/system-prompt/${mode}`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text }),
      });
    });
  });
}
```

Update the `DOMContentLoaded` listener at the bottom of the file:
```javascript
document.addEventListener("DOMContentLoaded", () => {
  loadSystemPrompts();
  setupSystemPromptForms();
  loadFamilies();
  setupFamilyForm();
  loadCharacters();
  setupCharacterForm();
});
```
(`loadFamilies`, `setupFamilyForm`, `loadCharacters`, `setupCharacterForm`, and everything else in the file are untouched.)

- [ ] **Step 3: Update `tests/test_pages.py`**

Replace `test_admin_page_has_management_sections`:
```python
def test_admin_page_has_management_sections(api_client, auth_headers):
    response = api_client.get("/admin", auth=auth_headers)

    assert response.status_code == 200
    assert 'id="family-form"' in response.text
    assert 'id="character-form"' in response.text
    assert 'id="system-prompt-generate-text"' in response.text
    assert 'id="system-prompt-iterate-text"' in response.text
    assert 'id="system-prompt-image-text"' in response.text
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
pytest tests/test_pages.py -v
```
Expected: PASS (all tests in the file).

- [ ] **Step 5: Run the full suite**

```bash
pytest -v
```
Expected: PASS (every test in the repo).

- [ ] **Step 6: Manually smoke-test in the browser**

Open `/admin`, confirm three separate system-prompt textareas (Generar/Iterar/Imagen) each load their own seeded default text and save independently (edit one, save, reload the page, confirm only that one changed).

- [ ] **Step 7: Commit**

```bash
git add app/templates/admin.html app/static/admin.js tests/test_pages.py
git commit -m "Split admin system prompt editor into three independent sections"
```

---

## Task 8: Documentation and Final Verification

**Files:**
- Modify: `README.md`
- Modify: `CLAUDE.md`
- No code changes — verification only otherwise.

**Interfaces:** exercises the full stack built in Tasks 1-7.

- [ ] **Step 1: Update `README.md`**

In the Features list, replace the separate "Generar"/"Iterar" bullets with one merged bullet, and update the Admin bullet to mention three prompts. Find and replace:
```markdown
- **Generar** — turn a natural-language idea into a ready-to-copy positive/negative prompt pair for the selected model family.
- **Iterar** — refine an existing prompt (generated here or pasted from elsewhere) with follow-up instructions.
```
with:
```markdown
- **Generar** — turn a natural-language idea into a ready-to-copy positive/negative prompt pair for the selected model family, or refine an existing prompt (generated here or pasted from elsewhere) by leaving the "existing prompt" field filled in.
```
And find:
```markdown
- **Admin panel** — edit the global system prompt, manage model families (their rules and whether they use negative prompts), and manage personajes, all from the browser.
```
Replace with:
```markdown
- **Admin panel** — edit the three independent system prompts (Generar/Iterar/Imagen), manage model families (their rules and whether they use negative prompts), and manage personajes, all from the browser.
```

- [ ] **Step 2: Update `CLAUDE.md`**

Three exact find-and-replace edits. If any of the "find" text doesn't match `CLAUDE.md` verbatim (it may have drifted since this plan was written), stop and re-read the file rather than guessing — do not force a fuzzy match.

**Edit 2a** — find this sentence inside the "Request flow" paragraph:
```
On startup, `lifespan()` in `main.py` calls `seed.ensure_seed_data(data_dir)`, which creates `families.json` (seeded with SDXL and Z-Image-Turbo, each with hardcoded `*_INSTRUCTIONS` text), an empty `characters.json`, and `system_prompt.txt` (seeded with `DEFAULT_GLOBAL_SYSTEM_PROMPT`) — but only if those files don't already exist, so it's a one-time bootstrap, not a sync.
```
Replace with:
```
On startup, `lifespan()` in `main.py` calls `seed.ensure_seed_data(data_dir)`, which creates `families.json` (seeded with SDXL and Z-Image-Turbo, each with hardcoded `*_INSTRUCTIONS` text), an empty `characters.json`, and `system_prompts.json` (seeded with three independent default prompts, one per mode: `generate`, `iterate`, `image`) — but only if those files don't already exist, so it's a one-time bootstrap, not a sync.
```

**Edit 2b** — find the entire `routes/api.py` paragraph:
```
**`routes/api.py`** is where generation actually happens, and all three modes (`/api/generate`, `/api/iterate`, `/api/from-image`) follow the same composition: look up the family via `families.get_family(family_id)` (404 if unknown) → build a system prompt with `prompts.build_system_prompt(system_prompt.read_system_prompt(), family, is_iteration=...)` (global prompt + family-specific instructions, plus an iteration addendum when refining) → build a user message with `prompts.build_user_message(mode, ...)` → call `openrouter_client.call_openrouter(...)` with the appropriate model (`llm_model` for generate/iterate, `vision_model` + `image_data_uri` for from-image) → parse the raw `POSITIVE:`/`NEGATIVE:` response via `prompts.parse_response(response, family["has_negative_prompt"])` → log everything via `history.append_entry(...)` → return `{positive_prompt, negative_prompt}`. The three modes differ only in what goes into the user message (`generate` sends the idea; `iterate` sends the previous prompt + requested changes and flips `is_iteration=True` so the system prompt gets `ITERATION_ADDENDUM` telling the model to preserve and only modify what's asked; `from-image` sends an idea-optional caption plus a base64 data-URI image, dispatched to a vision-capable OpenRouter model) and what's persisted to history (`previous_prompt` for iterate, `vision_model` for from-image).
```
Replace with:
```
**`routes/api.py`** is where generation actually happens, and there are two endpoints, not three: `POST /api/generate` handles both the "generate" and "iterate" modes (an optional `previous_prompt` field decides which — blank means generate-from-scratch, non-blank means iterate-on-that-prompt), and `POST /api/from-image` handles the vision mode. All three *modes* (`generate`/`iterate`/`image`) follow the same composition: look up the family via `families.get_family(family_id)` (404 if unknown) → build a system prompt with `prompts.build_system_prompt(mode, family)`, which internally reads that mode's independent prompt from `data/system_prompts.json` and appends the family's instructions → build a user message with `prompts.build_user_message(mode, ...)` → call `openrouter_client.call_openrouter(...)` with the appropriate model (`llm_model` for generate/iterate, `vision_model` + `image_data_uri` for from-image) → parse the raw `POSITIVE:`/`NEGATIVE:` response via `prompts.parse_response(response, family["has_negative_prompt"])` → log everything via `history.append_entry(...)` → return `{positive_prompt, negative_prompt}`. Unlike the other two modes, `generate` and `iterate` share a single route and Pydantic model — the route computes `mode` from `previous_prompt` before doing anything else, and everything downstream (system prompt, user message, history entry) is driven by that one string.
```

**Edit 2c** — find this sentence inside the "Frontend" paragraph:
```
`index.html`/`app.js` (three tabs — Generar/Iterar/Imagen — toggled by CSS class, each a plain form posting JSON or FormData to `/api/*`; "personajes" render as buttons that insert their text at the textarea cursor; a "Iterar este prompt" button hands the just-generated positive prompt to the Iterar tab), `admin.html`/`admin.js` (system prompt textarea + family/character list-and-form CRUD panels, all calling `/api/admin/*`),
```
Replace with:
```
`index.html`/`app.js` (two tabs — Generar (which itself covers both generate and iterate, via the "existing prompt" field) and Imagen — toggled by CSS class, each a plain form posting JSON or FormData to `/api/*`; "personajes" render as buttons that insert their text at the textarea cursor; a "Iterar este prompt" button fills the "existing prompt" field in the same Generar tab with the just-generated positive prompt), `admin.html`/`admin.js` (three independent system prompt textareas, one per mode, + family/character list-and-form CRUD panels, all calling `/api/admin/*`),
```

- [ ] **Step 3: Run the full automated test suite one last time**

```bash
pytest -v
```
Expected: all tests PASS, no warnings.

- [ ] **Step 4: Manually verify locally**

```bash
uvicorn app.main:app --reload
```
- Open `/`, confirm 2 tabs (Generar, Imagen).
- On Generar, leave "Prompt existente" empty, generate something — confirm it works and check `/historial` shows `mode: generate`.
- Click "Iterar este prompt" from the result, confirm the previous prompt lands in "Prompt existente" and you're still on the Generar tab; submit a change — confirm `/historial` shows a second entry with `mode: iterate` and the correct `previous_prompt`.
- On Imagen, generate from an uploaded image — confirm it still works and check `/historial` shows `mode: image`.
- On `/admin`, confirm the three system-prompt sections show distinct seeded text, edit one, save, reload, confirm it persisted and the other two are untouched.

- [ ] **Step 5: Commit the documentation updates**

```bash
git add README.md CLAUDE.md
git commit -m "Update README and CLAUDE.md for merged Generar/Iterar and per-mode system prompts"
```

---

## Post-Plan Follow-Up (not part of this plan's tasks)

The app is already deployed on `cubi-n` (see prior deployment). After this plan's tasks are complete and reviewed, redeploying there means: `git pull` in `~/prompt-enhancer`, then `docker compose up -d --build`. Because `data/system_prompt.txt` (old) is superseded by `data/system_prompts.json` (new) and `ensure_seed_data` only seeds `system_prompts.json` if it's absent, the existing deployment will get fresh default prompts for all three modes on first boot after the update — this is expected per this plan's Non-Goal (no migration of old customization, since none exists yet on that deployment either).

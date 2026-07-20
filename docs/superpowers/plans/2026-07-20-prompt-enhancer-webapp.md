# Prompt Enhancer Web App Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the ComfyUI custom node with a self-hosted, single-user FastAPI web app that generates, iterates, and derives-from-image AI art prompts via OpenRouter, with an admin panel for the global system prompt, model families, and reusable "character" text snippets.

**Architecture:** A single FastAPI process serves both server-rendered Jinja2 pages and a small JSON API under `/api`. Flat JSON/JSONL files under `data/` (mounted as a Docker volume) hold families, characters, the global system prompt, and history — no database. A single vanilla JS file per page drives tab switching and `fetch()`-based calls to the API. HTTP Basic Auth (one shared username/password) protects the entire app.

**Tech Stack:** Python 3.10+, FastAPI, Uvicorn, Jinja2, `requests` (OpenRouter HTTP calls), `python-multipart` (image upload), `python-dotenv` (config), `pytest` + `httpx` (tests, dev-only).

## Global Constraints

- Single shared user: HTTP Basic Auth protects every route in the app (main pages, API, admin) — see spec "Auth".
- No database — persistence is flat JSON files (`families.json`, `characters.json`, `system_prompt.txt`) and an append-only `history.jsonl` — see spec "Data Model".
- Uploaded images are never persisted to disk or referenced in history — see spec "Data Model" / "Non-Goals".
- Image uploads: only `image/jpeg`, `image/png`, `image/webp`; max size 10 MB — see spec "Request Flow".
- A family's `has_negative_prompt` boolean (admin-editable data) determines whether the backend forces `negative_prompt = ""` after parsing — replaces the old hardcoded model-name check — see spec "Data Model".
- No JS framework and no frontend build step — plain Jinja2 templates + vanilla JS — see spec "Architecture".
- The ComfyUI node is fully removed as part of this work (`prompt_enhancer_node.py`, root `__init__.py`, root `openrouter_client.py`, root `system_prompts.py`, `web/`) — see spec "Context".
- Python version floor: 3.10 (uses `X | None` type hints throughout).

---

## Task 1: Project Scaffold

**Files:**
- Delete: `prompt_enhancer_node.py`, `__init__.py` (repo root), `openrouter_client.py` (repo root), `system_prompts.py` (repo root), `web/prompt_enhancer.js`, `web/` (now empty)
- Create: `app/__init__.py`, `app/main.py`
- Create: `tests/__init__.py`, `tests/test_main.py`
- Modify: `pyproject.toml`, `requirements.txt`, `.gitignore`

**Interfaces:**
- Produces: `app.main.app` — a `fastapi.FastAPI` instance (no routes yet; later tasks add routes to it).

- [ ] **Step 1: Remove the ComfyUI node files**

```bash
git rm prompt_enhancer_node.py __init__.py openrouter_client.py system_prompts.py web/prompt_enhancer.js
rmdir web
```

- [ ] **Step 2: Rewrite `pyproject.toml`**

```toml
[project]
name = "prompt-enhancer"
description = "Self-hosted web app that uses an LLM (via OpenRouter) to craft image-generation prompts for SDXL, Z-Image-Turbo, and other model families."
version = "2.0.0"
license = "MIT"
requires-python = ">=3.10"
dependencies = [
    "fastapi>=0.115.0",
    "uvicorn[standard]>=0.30.0",
    "python-multipart>=0.0.9",
    "jinja2>=3.1.0",
    "python-dotenv>=1.0.0",
    "requests>=2.28.0",
]

[project.optional-dependencies]
dev = ["pytest>=8.0.0", "httpx>=0.27.0"]
```

- [ ] **Step 3: Rewrite `requirements.txt`**

```
fastapi>=0.115.0
uvicorn[standard]>=0.30.0
python-multipart>=0.0.9
jinja2>=3.1.0
python-dotenv>=1.0.0
requests>=2.28.0
```

- [ ] **Step 4: Add `data/` to `.gitignore`**

Add this line under the "OS junk" section (or its own section) of the existing `.gitignore`:

```
# App data (persisted via Docker volume)
data/
```

- [ ] **Step 5: Create the `app` package**

`app/__init__.py`:
```python
```
(empty file — marks `app` as a package)

`app/main.py`:
```python
from fastapi import FastAPI

app = FastAPI(title="Prompt Enhancer")
```

- [ ] **Step 6: Write the failing scaffold test**

`tests/__init__.py`:
```python
```
(empty file)

`tests/test_main.py`:
```python
from fastapi import FastAPI

from app.main import app


def test_app_is_a_fastapi_instance():
    assert isinstance(app, FastAPI)
```

- [ ] **Step 7: Install dependencies and run the test**

```bash
pip install -e ".[dev]"
pytest tests/test_main.py -v
```
Expected: PASS (`test_app_is_a_fastapi_instance`).

- [ ] **Step 8: Commit**

```bash
git add -A
git commit -m "Scaffold FastAPI app, remove ComfyUI node"
```

---

## Task 2: Config and Auth

**Files:**
- Create: `app/config.py`, `app/auth.py`
- Create: `tests/test_auth.py`
- Modify: `app/main.py`
- Create: `.env.example`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces:
  - `app.config.Settings` (dataclass: `openrouter_api_key: str`, `admin_username: str`, `admin_password: str`, `data_dir: str`).
  - `app.config.get_settings() -> Settings` — reads env vars fresh on every call (no caching), so tests can `monkeypatch.setenv(...)` per-test.
  - `app.auth.require_auth(credentials: HTTPBasicCredentials = Depends(HTTPBasic())) -> str` — returns the username on success, raises `HTTPException(401)` on failure. Used by every later task's routes via the app-level dependency wired in this task.

- [ ] **Step 1: Write `app/config.py`**

```python
import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    openrouter_api_key: str
    admin_username: str
    admin_password: str
    data_dir: str


def get_settings() -> Settings:
    return Settings(
        openrouter_api_key=os.environ.get("OPENROUTER_API_KEY", ""),
        admin_username=os.environ.get("ADMIN_USERNAME", "admin"),
        admin_password=os.environ.get("ADMIN_PASSWORD", ""),
        data_dir=os.environ.get("DATA_DIR", "data"),
    )
```

- [ ] **Step 2: Write the failing auth tests**

`tests/test_auth.py`:
```python
import pytest
from fastapi import HTTPException
from fastapi.security import HTTPBasicCredentials

from app.auth import require_auth


def test_require_auth_accepts_matching_credentials(monkeypatch):
    monkeypatch.setenv("ADMIN_USERNAME", "admin")
    monkeypatch.setenv("ADMIN_PASSWORD", "secret")
    creds = HTTPBasicCredentials(username="admin", password="secret")

    assert require_auth(creds) == "admin"


def test_require_auth_rejects_wrong_password(monkeypatch):
    monkeypatch.setenv("ADMIN_USERNAME", "admin")
    monkeypatch.setenv("ADMIN_PASSWORD", "secret")
    creds = HTTPBasicCredentials(username="admin", password="wrong")

    with pytest.raises(HTTPException) as exc_info:
        require_auth(creds)
    assert exc_info.value.status_code == 401


def test_require_auth_rejects_wrong_username(monkeypatch):
    monkeypatch.setenv("ADMIN_USERNAME", "admin")
    monkeypatch.setenv("ADMIN_PASSWORD", "secret")
    creds = HTTPBasicCredentials(username="someone-else", password="secret")

    with pytest.raises(HTTPException) as exc_info:
        require_auth(creds)
    assert exc_info.value.status_code == 401
```

- [ ] **Step 3: Run the tests to verify they fail**

```bash
pytest tests/test_auth.py -v
```
Expected: FAIL (`ModuleNotFoundError: No module named 'app.auth'`).

- [ ] **Step 4: Write `app/auth.py`**

```python
import secrets

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials

from .config import get_settings

security = HTTPBasic()


def require_auth(credentials: HTTPBasicCredentials = Depends(security)) -> str:
    settings = get_settings()
    username_ok = secrets.compare_digest(credentials.username, settings.admin_username)
    password_ok = secrets.compare_digest(credentials.password, settings.admin_password)
    if not (username_ok and password_ok):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials.username
```

- [ ] **Step 5: Run the tests to verify they pass**

```bash
pytest tests/test_auth.py -v
```
Expected: PASS (all 3 tests).

- [ ] **Step 6: Wire the auth dependency into the app**

Modify `app/main.py`:
```python
from fastapi import Depends, FastAPI

from .auth import require_auth

app = FastAPI(title="Prompt Enhancer", dependencies=[Depends(require_auth)])
```

- [ ] **Step 7: Run the full test suite to confirm nothing broke**

```bash
pytest -v
```
Expected: PASS (`test_app_is_a_fastapi_instance` plus the 3 new auth tests).

- [ ] **Step 8: Create `.env.example`**

```
OPENROUTER_API_KEY=sk-or-...
ADMIN_USERNAME=admin
ADMIN_PASSWORD=change-me
```

- [ ] **Step 9: Commit**

```bash
git add app/config.py app/auth.py app/main.py tests/test_auth.py .env.example
git commit -m "Add config loading and HTTP Basic Auth, wired app-wide"
```

---

## Task 3: Storage Helpers

**Files:**
- Create: `app/storage.py`
- Create: `tests/test_storage.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces:
  - `app.storage.read_json(path: str, default: object) -> object`
  - `app.storage.write_json(path: str, data: object) -> None` (atomic: writes to a temp file then `os.replace`)
  - `app.storage.append_jsonl(path: str, entry: dict) -> None`
  - `app.storage.read_jsonl(path: str) -> list[dict]`

- [ ] **Step 1: Write the failing storage tests**

`tests/test_storage.py`:
```python
import json

from app import storage


def test_read_json_returns_default_when_file_missing(tmp_path):
    path = str(tmp_path / "missing.json")

    result = storage.read_json(path, default=[])

    assert result == []


def test_write_json_then_read_json_roundtrips(tmp_path):
    path = str(tmp_path / "data.json")

    storage.write_json(path, {"a": 1})

    assert storage.read_json(path, default=None) == {"a": 1}


def test_write_json_creates_parent_directories(tmp_path):
    path = str(tmp_path / "nested" / "dir" / "data.json")

    storage.write_json(path, [1, 2, 3])

    assert json.loads((tmp_path / "nested" / "dir" / "data.json").read_text()) == [1, 2, 3]


def test_write_json_does_not_leave_temp_files(tmp_path):
    path = str(tmp_path / "data.json")

    storage.write_json(path, {"a": 1})

    leftover = [p for p in tmp_path.iterdir() if p.name != "data.json"]
    assert leftover == []


def test_append_jsonl_then_read_jsonl_roundtrips(tmp_path):
    path = str(tmp_path / "history.jsonl")

    storage.append_jsonl(path, {"id": 1})
    storage.append_jsonl(path, {"id": 2})

    assert storage.read_jsonl(path) == [{"id": 1}, {"id": 2}]


def test_read_jsonl_returns_empty_list_when_file_missing(tmp_path):
    path = str(tmp_path / "missing.jsonl")

    assert storage.read_jsonl(path) == []
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
pytest tests/test_storage.py -v
```
Expected: FAIL (`ModuleNotFoundError: No module named 'app.storage'`).

- [ ] **Step 3: Write `app/storage.py`**

```python
import json
import os
import tempfile
from pathlib import Path
from typing import Any


def read_json(path: str, default: Any) -> Any:
    p = Path(path)
    if not p.exists():
        return default
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str, data: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=str(p.parent), prefix=".tmp-", suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        os.replace(tmp_path, p)
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


def append_jsonl(path: str, entry: dict) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def read_jsonl(path: str) -> list:
    p = Path(path)
    if not p.exists():
        return []
    entries = []
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
pytest tests/test_storage.py -v
```
Expected: PASS (all 6 tests).

- [ ] **Step 5: Commit**

```bash
git add app/storage.py tests/test_storage.py
git commit -m "Add atomic JSON and JSONL storage helpers"
```

---

## Task 4: Families Module

**Files:**
- Create: `app/families.py`
- Create: `tests/test_families.py`

**Interfaces:**
- Consumes: `app.storage.read_json`, `app.storage.write_json` (Task 3); `app.config.get_settings` (Task 2).
- Produces:
  - `app.families.list_families(path: str | None = None) -> list[dict]`
  - `app.families.get_family(family_id: str, path: str | None = None) -> dict | None`
  - `app.families.create_family(name: str, instructions: str, has_negative_prompt: bool, path: str | None = None) -> dict`
  - `app.families.update_family(family_id: str, name: str, instructions: str, has_negative_prompt: bool, path: str | None = None) -> dict | None`
  - `app.families.delete_family(family_id: str, path: str | None = None) -> bool`
  - Family dict shape: `{"id": str, "name": str, "instructions": str, "has_negative_prompt": bool}`.

- [ ] **Step 1: Write the failing families tests**

`tests/test_families.py`:
```python
from app import families


def test_list_families_empty_when_file_missing(tmp_path):
    path = str(tmp_path / "families.json")

    assert families.list_families(path=path) == []


def test_create_family_persists_and_is_listed(tmp_path):
    path = str(tmp_path / "families.json")

    created = families.create_family("SDXL", "some rules", True, path=path)

    assert created["name"] == "SDXL"
    assert created["instructions"] == "some rules"
    assert created["has_negative_prompt"] is True
    assert "id" in created
    assert families.list_families(path=path) == [created]


def test_get_family_returns_none_when_not_found(tmp_path):
    path = str(tmp_path / "families.json")
    families.create_family("SDXL", "rules", True, path=path)

    assert families.get_family("nonexistent-id", path=path) is None


def test_get_family_returns_matching_family(tmp_path):
    path = str(tmp_path / "families.json")
    created = families.create_family("SDXL", "rules", True, path=path)

    assert families.get_family(created["id"], path=path) == created


def test_update_family_changes_fields(tmp_path):
    path = str(tmp_path / "families.json")
    created = families.create_family("SDXL", "rules", True, path=path)

    updated = families.update_family(
        created["id"], "SDXL v2", "new rules", False, path=path
    )

    assert updated["name"] == "SDXL v2"
    assert updated["instructions"] == "new rules"
    assert updated["has_negative_prompt"] is False
    assert updated["id"] == created["id"]


def test_update_family_returns_none_when_not_found(tmp_path):
    path = str(tmp_path / "families.json")

    assert families.update_family("nonexistent-id", "x", "y", True, path=path) is None


def test_delete_family_removes_it(tmp_path):
    path = str(tmp_path / "families.json")
    created = families.create_family("SDXL", "rules", True, path=path)

    result = families.delete_family(created["id"], path=path)

    assert result is True
    assert families.list_families(path=path) == []


def test_delete_family_returns_false_when_not_found(tmp_path):
    path = str(tmp_path / "families.json")

    assert families.delete_family("nonexistent-id", path=path) is False
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
pytest tests/test_families.py -v
```
Expected: FAIL (`ModuleNotFoundError: No module named 'app.families'`).

- [ ] **Step 3: Write `app/families.py`**

```python
import uuid

from . import storage
from .config import get_settings


def _default_path() -> str:
    return f"{get_settings().data_dir}/families.json"


def list_families(path: str | None = None) -> list[dict]:
    return storage.read_json(path or _default_path(), default=[])


def get_family(family_id: str, path: str | None = None) -> dict | None:
    for family in list_families(path):
        if family["id"] == family_id:
            return family
    return None


def create_family(
    name: str, instructions: str, has_negative_prompt: bool, path: str | None = None
) -> dict:
    resolved_path = path or _default_path()
    all_families = list_families(resolved_path)
    family = {
        "id": str(uuid.uuid4()),
        "name": name,
        "instructions": instructions,
        "has_negative_prompt": has_negative_prompt,
    }
    all_families.append(family)
    storage.write_json(resolved_path, all_families)
    return family


def update_family(
    family_id: str,
    name: str,
    instructions: str,
    has_negative_prompt: bool,
    path: str | None = None,
) -> dict | None:
    resolved_path = path or _default_path()
    all_families = list_families(resolved_path)
    for family in all_families:
        if family["id"] == family_id:
            family["name"] = name
            family["instructions"] = instructions
            family["has_negative_prompt"] = has_negative_prompt
            storage.write_json(resolved_path, all_families)
            return family
    return None


def delete_family(family_id: str, path: str | None = None) -> bool:
    resolved_path = path or _default_path()
    all_families = list_families(resolved_path)
    remaining = [f for f in all_families if f["id"] != family_id]
    if len(remaining) == len(all_families):
        return False
    storage.write_json(resolved_path, remaining)
    return True
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
pytest tests/test_families.py -v
```
Expected: PASS (all 8 tests).

- [ ] **Step 5: Commit**

```bash
git add app/families.py tests/test_families.py
git commit -m "Add families CRUD module"
```

---

## Task 5: Characters Module

**Files:**
- Create: `app/characters.py`
- Create: `tests/test_characters.py`

**Interfaces:**
- Consumes: `app.storage` (Task 3), `app.config.get_settings` (Task 2).
- Produces:
  - `app.characters.list_characters(path: str | None = None) -> list[dict]`
  - `app.characters.create_character(name: str, text: str, path: str | None = None) -> dict`
  - `app.characters.update_character(character_id: str, name: str, text: str, path: str | None = None) -> dict | None`
  - `app.characters.delete_character(character_id: str, path: str | None = None) -> bool`
  - Character dict shape: `{"id": str, "name": str, "text": str}`.

- [ ] **Step 1: Write the failing characters tests**

`tests/test_characters.py`:
```python
from app import characters


def test_list_characters_empty_when_file_missing(tmp_path):
    path = str(tmp_path / "characters.json")

    assert characters.list_characters(path=path) == []


def test_create_character_persists_and_is_listed(tmp_path):
    path = str(tmp_path / "characters.json")

    created = characters.create_character("Warrior", "a fierce warrior", path=path)

    assert created["name"] == "Warrior"
    assert created["text"] == "a fierce warrior"
    assert "id" in created
    assert characters.list_characters(path=path) == [created]


def test_update_character_changes_fields(tmp_path):
    path = str(tmp_path / "characters.json")
    created = characters.create_character("Warrior", "a fierce warrior", path=path)

    updated = characters.update_character(
        created["id"], "Warrior v2", "an even fiercer warrior", path=path
    )

    assert updated["name"] == "Warrior v2"
    assert updated["text"] == "an even fiercer warrior"
    assert updated["id"] == created["id"]


def test_update_character_returns_none_when_not_found(tmp_path):
    path = str(tmp_path / "characters.json")

    assert characters.update_character("nonexistent-id", "x", "y", path=path) is None


def test_delete_character_removes_it(tmp_path):
    path = str(tmp_path / "characters.json")
    created = characters.create_character("Warrior", "a fierce warrior", path=path)

    result = characters.delete_character(created["id"], path=path)

    assert result is True
    assert characters.list_characters(path=path) == []


def test_delete_character_returns_false_when_not_found(tmp_path):
    path = str(tmp_path / "characters.json")

    assert characters.delete_character("nonexistent-id", path=path) is False
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
pytest tests/test_characters.py -v
```
Expected: FAIL (`ModuleNotFoundError: No module named 'app.characters'`).

- [ ] **Step 3: Write `app/characters.py`**

```python
import uuid

from . import storage
from .config import get_settings


def _default_path() -> str:
    return f"{get_settings().data_dir}/characters.json"


def list_characters(path: str | None = None) -> list[dict]:
    return storage.read_json(path or _default_path(), default=[])


def create_character(name: str, text: str, path: str | None = None) -> dict:
    resolved_path = path or _default_path()
    all_characters = list_characters(resolved_path)
    character = {"id": str(uuid.uuid4()), "name": name, "text": text}
    all_characters.append(character)
    storage.write_json(resolved_path, all_characters)
    return character


def update_character(
    character_id: str, name: str, text: str, path: str | None = None
) -> dict | None:
    resolved_path = path or _default_path()
    all_characters = list_characters(resolved_path)
    for character in all_characters:
        if character["id"] == character_id:
            character["name"] = name
            character["text"] = text
            storage.write_json(resolved_path, all_characters)
            return character
    return None


def delete_character(character_id: str, path: str | None = None) -> bool:
    resolved_path = path or _default_path()
    all_characters = list_characters(resolved_path)
    remaining = [c for c in all_characters if c["id"] != character_id]
    if len(remaining) == len(all_characters):
        return False
    storage.write_json(resolved_path, remaining)
    return True
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
pytest tests/test_characters.py -v
```
Expected: PASS (all 6 tests).

- [ ] **Step 5: Commit**

```bash
git add app/characters.py tests/test_characters.py
git commit -m "Add characters CRUD module"
```

---

## Task 6: History Module

**Files:**
- Create: `app/history.py`
- Create: `tests/test_history.py`

**Interfaces:**
- Consumes: `app.storage.append_jsonl`, `app.storage.read_jsonl` (Task 3); `app.config.get_settings` (Task 2).
- Produces:
  - `app.history.append_entry(mode: str, family_id: str, family_name: str, llm_model: str | None, vision_model: str | None, temperature: float, user_input: str, example_prompts: str, previous_prompt: str | None, positive_prompt: str, negative_prompt: str, path: str | None = None) -> dict`
  - `app.history.list_entries(path: str | None = None) -> list[dict]` — newest first.

- [ ] **Step 1: Write the failing history tests**

`tests/test_history.py`:
```python
from app import history


def test_list_entries_empty_when_file_missing(tmp_path):
    path = str(tmp_path / "history.jsonl")

    assert history.list_entries(path=path) == []


def test_append_entry_persists_all_fields(tmp_path):
    path = str(tmp_path / "history.jsonl")

    entry = history.append_entry(
        mode="generate",
        family_id="fam-1",
        family_name="SDXL",
        llm_model="anthropic/claude-sonnet-4",
        vision_model=None,
        temperature=0.7,
        user_input="a cyberpunk samurai",
        example_prompts="",
        previous_prompt=None,
        positive_prompt="a cyberpunk samurai in neon rain",
        negative_prompt="blurry, deformed hands",
        path=path,
    )

    assert entry["mode"] == "generate"
    assert entry["family_name"] == "SDXL"
    assert entry["positive_prompt"] == "a cyberpunk samurai in neon rain"
    assert "id" in entry
    assert "timestamp" in entry


def test_list_entries_returns_newest_first(tmp_path):
    path = str(tmp_path / "history.jsonl")
    first = history.append_entry(
        mode="generate", family_id="f", family_name="SDXL", llm_model="m",
        vision_model=None, temperature=0.7, user_input="first", example_prompts="",
        previous_prompt=None, positive_prompt="first result", negative_prompt="",
        path=path,
    )
    second = history.append_entry(
        mode="generate", family_id="f", family_name="SDXL", llm_model="m",
        vision_model=None, temperature=0.7, user_input="second", example_prompts="",
        previous_prompt=None, positive_prompt="second result", negative_prompt="",
        path=path,
    )

    entries = history.list_entries(path=path)

    assert [e["id"] for e in entries] == [second["id"], first["id"]]
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
pytest tests/test_history.py -v
```
Expected: FAIL (`ModuleNotFoundError: No module named 'app.history'`).

- [ ] **Step 3: Write `app/history.py`**

```python
import uuid
from datetime import datetime, timezone

from . import storage
from .config import get_settings


def _default_path() -> str:
    return f"{get_settings().data_dir}/history.jsonl"


def append_entry(
    mode: str,
    family_id: str,
    family_name: str,
    llm_model: str | None,
    vision_model: str | None,
    temperature: float,
    user_input: str,
    example_prompts: str,
    previous_prompt: str | None,
    positive_prompt: str,
    negative_prompt: str,
    path: str | None = None,
) -> dict:
    entry = {
        "id": str(uuid.uuid4()),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "mode": mode,
        "family_id": family_id,
        "family_name": family_name,
        "llm_model": llm_model,
        "vision_model": vision_model,
        "temperature": temperature,
        "user_input": user_input,
        "example_prompts": example_prompts,
        "previous_prompt": previous_prompt,
        "positive_prompt": positive_prompt,
        "negative_prompt": negative_prompt,
    }
    storage.append_jsonl(path or _default_path(), entry)
    return entry


def list_entries(path: str | None = None) -> list[dict]:
    entries = storage.read_jsonl(path or _default_path())
    return list(reversed(entries))
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
pytest tests/test_history.py -v
```
Expected: PASS (all 3 tests).

- [ ] **Step 5: Commit**

```bash
git add app/history.py tests/test_history.py
git commit -m "Add append-only history module"
```

---

## Task 7: System Prompt Module

**Files:**
- Create: `app/system_prompt.py`
- Create: `tests/test_system_prompt.py`

**Interfaces:**
- Consumes: `app.config.get_settings` (Task 2).
- Produces:
  - `app.system_prompt.read_system_prompt(path: str | None = None) -> str` (returns `""` if the file doesn't exist)
  - `app.system_prompt.write_system_prompt(text: str, path: str | None = None) -> None`

- [ ] **Step 1: Write the failing system prompt tests**

`tests/test_system_prompt.py`:
```python
from app import system_prompt


def test_read_system_prompt_returns_empty_string_when_missing(tmp_path):
    path = str(tmp_path / "system_prompt.txt")

    assert system_prompt.read_system_prompt(path=path) == ""


def test_write_then_read_roundtrips(tmp_path):
    path = str(tmp_path / "system_prompt.txt")

    system_prompt.write_system_prompt("You are an expert prompt engineer.", path=path)

    assert system_prompt.read_system_prompt(path=path) == "You are an expert prompt engineer."


def test_write_creates_parent_directories(tmp_path):
    path = str(tmp_path / "nested" / "system_prompt.txt")

    system_prompt.write_system_prompt("hello", path=path)

    assert (tmp_path / "nested" / "system_prompt.txt").read_text(encoding="utf-8") == "hello"
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
pytest tests/test_system_prompt.py -v
```
Expected: FAIL (`ModuleNotFoundError: No module named 'app.system_prompt'`).

- [ ] **Step 3: Write `app/system_prompt.py`**

```python
from pathlib import Path

from .config import get_settings


def _default_path() -> str:
    return f"{get_settings().data_dir}/system_prompt.txt"


def read_system_prompt(path: str | None = None) -> str:
    p = Path(path or _default_path())
    return p.read_text(encoding="utf-8") if p.exists() else ""


def write_system_prompt(text: str, path: str | None = None) -> None:
    p = Path(path or _default_path())
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
pytest tests/test_system_prompt.py -v
```
Expected: PASS (all 3 tests).

- [ ] **Step 5: Commit**

```bash
git add app/system_prompt.py tests/test_system_prompt.py
git commit -m "Add global system prompt read/write module"
```

---

## Task 8: Seed Data Bootstrap

**Files:**
- Create: `app/seed.py`
- Create: `tests/test_seed.py`

**Interfaces:**
- Consumes: `app.families.create_family` (Task 4), `app.storage.write_json` (Task 3), `app.system_prompt.write_system_prompt` (Task 7).
- Produces: `app.seed.ensure_seed_data(data_dir: str) -> None` — idempotent; only writes files that don't already exist.

This ports the existing SDXL/ZIT prompting rules from the (now-deleted) `system_prompts.py` into seed content for the new admin-managed `families.json`.

- [ ] **Step 1: Write the failing seed tests**

`tests/test_seed.py`:
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


def test_ensure_seed_data_creates_system_prompt_file(tmp_path):
    ensure_seed_data(str(tmp_path))

    text = (tmp_path / "system_prompt.txt").read_text(encoding="utf-8")
    assert "POSITIVE:" in text
    assert "NEGATIVE:" in text


def test_ensure_seed_data_is_idempotent(tmp_path):
    ensure_seed_data(str(tmp_path))
    ensure_seed_data(str(tmp_path))

    families = json.loads((tmp_path / "families.json").read_text())
    assert len(families) == 2


def test_ensure_seed_data_does_not_overwrite_existing_system_prompt(tmp_path):
    (tmp_path / "system_prompt.txt").write_text("custom prompt", encoding="utf-8")

    ensure_seed_data(str(tmp_path))

    assert (tmp_path / "system_prompt.txt").read_text(encoding="utf-8") == "custom prompt"
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
pytest tests/test_seed.py -v
```
Expected: FAIL (`ModuleNotFoundError: No module named 'app.seed'`).

- [ ] **Step 3: Write `app/seed.py`**

```python
from pathlib import Path

from . import families, storage, system_prompt

DEFAULT_GLOBAL_SYSTEM_PROMPT = """You are an expert AI image-generation prompt engineer. Transform the user's idea (or requested changes to an existing prompt, or an attached reference image) into a high-quality prompt for the selected model family, following that family's specific rules exactly.

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
    system_prompt_path = f"{data_dir}/system_prompt.txt"

    if not Path(families_path).exists():
        families.create_family("SDXL", SDXL_INSTRUCTIONS, True, path=families_path)
        families.create_family("Z-Image-Turbo", ZIT_INSTRUCTIONS, False, path=families_path)

    if not Path(characters_path).exists():
        storage.write_json(characters_path, [])

    if not Path(system_prompt_path).exists():
        system_prompt.write_system_prompt(DEFAULT_GLOBAL_SYSTEM_PROMPT, path=system_prompt_path)
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
pytest tests/test_seed.py -v
```
Expected: PASS (all 5 tests).

- [ ] **Step 5: Commit**

```bash
git add app/seed.py tests/test_seed.py
git commit -m "Add seed data bootstrap for families, characters, and system prompt"
```

---

## Task 9: OpenRouter Client

**Files:**
- Create: `app/openrouter_client.py`
- Create: `tests/test_openrouter_client.py`

**Interfaces:**
- Consumes: nothing from earlier tasks (standalone HTTP wrapper).
- Produces: `app.openrouter_client.call_openrouter(api_key: str, model: str, system_prompt: str, user_message: str, temperature: float = 0.7, image_data_uri: str | None = None) -> str` — raises `RuntimeError` with a human-readable message on any failure.

- [ ] **Step 1: Write the failing openrouter client tests**

`tests/test_openrouter_client.py`:
```python
from unittest.mock import Mock, patch

import pytest
import requests

from app.openrouter_client import call_openrouter


def _mock_response(status_code=200, json_data=None, text=""):
    response = Mock()
    response.status_code = status_code
    response.json.return_value = json_data or {}
    response.text = text
    return response


@patch("app.openrouter_client.requests.post")
def test_call_openrouter_returns_message_content(mock_post):
    mock_post.return_value = _mock_response(
        200, {"choices": [{"message": {"content": "POSITIVE: a cat\nNEGATIVE: blurry"}}]}
    )

    result = call_openrouter(
        api_key="key", model="anthropic/claude-sonnet-4",
        system_prompt="sys", user_message="user", temperature=0.7,
    )

    assert result == "POSITIVE: a cat\nNEGATIVE: blurry"


@patch("app.openrouter_client.requests.post")
def test_call_openrouter_sends_plain_string_content_without_image(mock_post):
    mock_post.return_value = _mock_response(200, {"choices": [{"message": {"content": "ok"}}]})

    call_openrouter(api_key="key", model="m", system_prompt="sys", user_message="hello")

    sent_payload = mock_post.call_args.kwargs["json"]
    assert sent_payload["messages"][1]["content"] == "hello"


@patch("app.openrouter_client.requests.post")
def test_call_openrouter_sends_multimodal_content_with_image(mock_post):
    mock_post.return_value = _mock_response(200, {"choices": [{"message": {"content": "ok"}}]})

    call_openrouter(
        api_key="key", model="m", system_prompt="sys", user_message="hello",
        image_data_uri="data:image/png;base64,AAAA",
    )

    sent_payload = mock_post.call_args.kwargs["json"]
    content = sent_payload["messages"][1]["content"]
    assert content == [
        {"type": "text", "text": "hello"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
    ]


@patch("app.openrouter_client.requests.post")
def test_call_openrouter_raises_on_timeout(mock_post):
    mock_post.side_effect = requests.exceptions.Timeout()

    with pytest.raises(RuntimeError, match="timed out"):
        call_openrouter(api_key="key", model="m", system_prompt="s", user_message="u")


@patch("app.openrouter_client.requests.post")
def test_call_openrouter_raises_on_401(mock_post):
    mock_post.return_value = _mock_response(401)

    with pytest.raises(RuntimeError, match="invalid"):
        call_openrouter(api_key="key", model="m", system_prompt="s", user_message="u")


@patch("app.openrouter_client.requests.post")
def test_call_openrouter_raises_on_402(mock_post):
    mock_post.return_value = _mock_response(402)

    with pytest.raises(RuntimeError, match="credits"):
        call_openrouter(api_key="key", model="m", system_prompt="s", user_message="u")


@patch("app.openrouter_client.requests.post")
def test_call_openrouter_raises_on_429(mock_post):
    mock_post.return_value = _mock_response(429)

    with pytest.raises(RuntimeError, match="rate limit"):
        call_openrouter(api_key="key", model="m", system_prompt="s", user_message="u")


@patch("app.openrouter_client.requests.post")
def test_call_openrouter_raises_on_malformed_response(mock_post):
    mock_post.return_value = _mock_response(200, {"unexpected": "shape"})

    with pytest.raises(RuntimeError, match="Unexpected response format"):
        call_openrouter(api_key="key", model="m", system_prompt="s", user_message="u")
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
pytest tests/test_openrouter_client.py -v
```
Expected: FAIL (`ModuleNotFoundError: No module named 'app.openrouter_client'`).

- [ ] **Step 3: Write `app/openrouter_client.py`**

```python
import requests

OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
TIMEOUT_SECONDS = 60


def call_openrouter(
    api_key: str,
    model: str,
    system_prompt: str,
    user_message: str,
    temperature: float = 0.7,
    image_data_uri: str | None = None,
) -> str:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    if image_data_uri:
        user_content = [
            {"type": "text", "text": user_message},
            {"type": "image_url", "image_url": {"url": image_data_uri}},
        ]
    else:
        user_content = user_message

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        "temperature": temperature,
        "max_tokens": 1024,
    }

    try:
        response = requests.post(
            OPENROUTER_API_URL, headers=headers, json=payload, timeout=TIMEOUT_SECONDS
        )
    except requests.exceptions.Timeout:
        raise RuntimeError(
            f"OpenRouter request timed out after {TIMEOUT_SECONDS}s. "
            "The LLM may be overloaded -- try again."
        )
    except requests.exceptions.ConnectionError:
        raise RuntimeError(
            "Could not connect to OpenRouter. Check your internet connection."
        )
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Network error calling OpenRouter: {e}")

    if response.status_code == 401:
        raise RuntimeError(
            "OpenRouter API key is invalid or missing. Check the OPENROUTER_API_KEY "
            "server configuration."
        )
    if response.status_code == 402:
        raise RuntimeError(
            "OpenRouter account has insufficient credits. "
            "Add credits at https://openrouter.ai/credits"
        )
    if response.status_code == 429:
        raise RuntimeError(
            "OpenRouter rate limit exceeded. Wait a moment and try again."
        )
    if response.status_code != 200:
        raise RuntimeError(
            f"OpenRouter returned HTTP {response.status_code}: {response.text[:300]}"
        )

    data = response.json()
    try:
        return data["choices"][0]["message"]["content"]
    except (KeyError, IndexError):
        raise RuntimeError(
            f"Unexpected response format from OpenRouter: {str(data)[:300]}"
        )
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
pytest tests/test_openrouter_client.py -v
```
Expected: PASS (all 9 tests).

- [ ] **Step 5: Commit**

```bash
git add app/openrouter_client.py tests/test_openrouter_client.py
git commit -m "Add OpenRouter client with text and multimodal support"
```

---

## Task 10: Prompts Module

**Files:**
- Create: `app/prompts.py`
- Create: `tests/test_prompts.py`

**Interfaces:**
- Consumes: nothing from earlier tasks (pure string logic).
- Produces:
  - `app.prompts.build_system_prompt(global_system_prompt: str, family: dict, is_iteration: bool = False) -> str`
  - `app.prompts.build_user_message(mode: str, user_input: str, previous_prompt: str | None = None, example_prompts: str | None = None) -> str` — `mode` is one of `"generate"`, `"iterate"`, `"image"`.
  - `app.prompts.parse_response(response: str, has_negative_prompt: bool) -> tuple[str, str]`

- [ ] **Step 1: Write the failing prompts tests**

`tests/test_prompts.py`:
```python
from app.prompts import build_system_prompt, build_user_message, parse_response


def test_build_system_prompt_combines_global_and_family():
    family = {"instructions": "family rules"}

    result = build_system_prompt("global rules", family, is_iteration=False)

    assert result == "global rules\n\nfamily rules"


def test_build_system_prompt_appends_iteration_addendum():
    family = {"instructions": "family rules"}

    result = build_system_prompt("global rules", family, is_iteration=True)

    assert result.startswith("global rules\n\nfamily rules")
    assert "Iteration Mode" in result
    assert "PRESERVE the good elements" in result


def test_build_user_message_generate_mode():
    result = build_user_message("generate", "a cyberpunk samurai")

    assert result == "## Idea\na cyberpunk samurai"


def test_build_user_message_generate_mode_with_example_prompts():
    result = build_user_message("generate", "a cyberpunk samurai", example_prompts="cinematic, moody")

    assert result == "## Idea\na cyberpunk samurai\n\n## Prompts de Ejemplo\ncinematic, moody"


def test_build_user_message_iterate_mode():
    result = build_user_message("iterate", "add lightning", previous_prompt="a samurai in rain")

    assert result == "## Prompt Previo\na samurai in rain\n\n## Cambios Solicitados\nadd lightning"


def test_build_user_message_image_mode_without_extra_text():
    result = build_user_message("image", "")

    assert result == "## Imagen\nSe adjunta una imagen de referencia para este prompt."


def test_build_user_message_image_mode_with_extra_text():
    result = build_user_message("image", "make it night time")

    assert result == (
        "## Imagen\nSe adjunta una imagen de referencia para este prompt.\n\n"
        "## Idea\nmake it night time"
    )


def test_parse_response_with_markers():
    response = "POSITIVE: a cyberpunk samurai\nNEGATIVE: blurry, extra fingers"

    positive, negative = parse_response(response, has_negative_prompt=True)

    assert positive == "a cyberpunk samurai"
    assert negative == "blurry, extra fingers"


def test_parse_response_forces_empty_negative_when_family_does_not_use_one():
    response = "POSITIVE: a flowing scene\nNEGATIVE: blurry"

    positive, negative = parse_response(response, has_negative_prompt=False)

    assert positive == "a flowing scene"
    assert negative == ""


def test_parse_response_falls_back_when_markers_missing():
    response = "Sure, here's your prompt:\na cyberpunk samurai in neon rain\nHope this helps!"

    positive, negative = parse_response(response, has_negative_prompt=True)

    assert positive == "a cyberpunk samurai in neon rain"
    assert negative == ""
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
pytest tests/test_prompts.py -v
```
Expected: FAIL (`ModuleNotFoundError: No module named 'app.prompts'`).

- [ ] **Step 3: Write `app/prompts.py`**

```python
import re

ITERATION_ADDENDUM = """

## Iteration Mode
The user is refining an EXISTING prompt. You will receive:
1. The previous prompt that was already generated
2. The user's requested changes

Your job is to:
- PRESERVE the good elements from the previous prompt -- do not regenerate from scratch
- Apply the user's requested changes precisely
- Maintain the same overall style and structure
- Only modify what the user explicitly asks to change
- If the user asks to "add" something, integrate it naturally into the existing prompt
- If the user asks to "remove" something, take it out cleanly without leaving gaps
"""


def build_system_prompt(global_system_prompt: str, family: dict, is_iteration: bool = False) -> str:
    prompt = f"{global_system_prompt.strip()}\n\n{family['instructions'].strip()}"
    if is_iteration:
        prompt += ITERATION_ADDENDUM
    return prompt


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

- [ ] **Step 4: Run the tests to verify they pass**

```bash
pytest tests/test_prompts.py -v
```
Expected: PASS (all 10 tests).

- [ ] **Step 5: Commit**

```bash
git add app/prompts.py tests/test_prompts.py
git commit -m "Add system prompt assembly, user message building, and response parsing"
```

---

## Task 11: API Route — `POST /api/generate`

**Files:**
- Create: `app/routes/__init__.py`, `app/routes/api.py`
- Create: `tests/conftest.py`, `tests/test_api_generate.py`
- Modify: `app/main.py`

**Interfaces:**
- Consumes: `app.families.get_family` (Task 4), `app.history.append_entry` (Task 6), `app.system_prompt.read_system_prompt` (Task 7), `app.openrouter_client.call_openrouter` (Task 9), `app.prompts.build_system_prompt` / `build_user_message` / `parse_response` (Task 10), `app.config.get_settings` (Task 2).
- Produces:
  - `app.routes.api.router` — a `fastapi.APIRouter` mounted at `/api`, included into `app.main.app` in this task.
  - `tests/conftest.py` fixtures: `api_client` (a `TestClient` with `DATA_DIR`/`ADMIN_USERNAME`/`ADMIN_PASSWORD`/`OPENROUTER_API_KEY` env vars monkeypatched to a `tmp_path`-backed sandbox) and `auth_headers` (the matching `(username, password)` tuple for `auth=` in requests). Later API-route tasks reuse these fixtures.

- [ ] **Step 1: Write `tests/conftest.py`**

```python
import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture
def api_client(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("ADMIN_USERNAME", "admin")
    monkeypatch.setenv("ADMIN_PASSWORD", "test-password")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    return TestClient(app)


@pytest.fixture
def auth_headers():
    return ("admin", "test-password")
```

- [ ] **Step 2: Write the failing `/api/generate` tests**

`tests/test_api_generate.py`:
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
def test_generate_appends_to_history(mock_call, api_client, auth_headers, tmp_path):
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
```

- [ ] **Step 3: Run the tests to verify they fail**

```bash
pytest tests/test_api_generate.py -v
```
Expected: FAIL (`ModuleNotFoundError: No module named 'app.routes'`).

- [ ] **Step 4: Write `app/routes/__init__.py`**

```python
```
(empty file)

- [ ] **Step 5: Write `app/routes/api.py`**

```python
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
```

- [ ] **Step 6: Wire the router into the app**

Modify `app/main.py`:
```python
from fastapi import Depends, FastAPI

from .auth import require_auth
from .routes import api as api_routes

app = FastAPI(title="Prompt Enhancer", dependencies=[Depends(require_auth)])

app.include_router(api_routes.router)
```

- [ ] **Step 7: Run the tests to verify they pass**

```bash
pytest tests/test_api_generate.py -v
```
Expected: PASS (all 6 tests).

- [ ] **Step 8: Run the full suite**

```bash
pytest -v
```
Expected: PASS (all tests so far).

- [ ] **Step 9: Commit**

```bash
git add app/routes/__init__.py app/routes/api.py app/main.py tests/conftest.py tests/test_api_generate.py
git commit -m "Add POST /api/generate endpoint"
```

---

## Task 12: API Route — `POST /api/iterate`

**Files:**
- Modify: `app/routes/api.py`
- Create: `tests/test_api_iterate.py`

**Interfaces:**
- Consumes: same as Task 11, plus `prompts.build_system_prompt(..., is_iteration=True)` and `prompts.build_user_message("iterate", ...)`.
- Produces: adds `POST /api/iterate` to `app.routes.api.router`.

- [ ] **Step 1: Write the failing `/api/iterate` tests**

`tests/test_api_iterate.py`:
```python
from unittest.mock import patch

from app import families


def _create_family(tmp_path, has_negative_prompt=True):
    return families.create_family(
        "SDXL", "family instructions", has_negative_prompt,
        path=str(tmp_path / "families.json"),
    )


def test_iterate_requires_auth(api_client):
    response = api_client.post("/api/iterate", json={
        "user_input": "add lightning", "previous_prompt": "a samurai",
        "family_id": "x", "llm_model": "m",
    })

    assert response.status_code == 401


@patch("app.routes.api.call_openrouter")
def test_iterate_returns_parsed_prompt(mock_call, api_client, auth_headers, tmp_path):
    family = _create_family(tmp_path)
    mock_call.return_value = "POSITIVE: a samurai with lightning\nNEGATIVE: blurry"

    response = api_client.post(
        "/api/iterate",
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


def test_iterate_rejects_blank_previous_prompt(api_client, auth_headers, tmp_path):
    family = _create_family(tmp_path)

    response = api_client.post(
        "/api/iterate",
        json={
            "user_input": "add lightning", "previous_prompt": "  ",
            "family_id": family["id"], "llm_model": "m",
        },
        auth=auth_headers,
    )

    assert response.status_code == 400


@patch("app.routes.api.call_openrouter")
def test_iterate_appends_to_history_with_previous_prompt(mock_call, api_client, auth_headers, tmp_path):
    family = _create_family(tmp_path)
    mock_call.return_value = "POSITIVE: updated\nNEGATIVE: "

    api_client.post(
        "/api/iterate",
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
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
pytest tests/test_api_iterate.py -v
```
Expected: FAIL (`404 Not Found` for `/api/iterate`, since the route doesn't exist yet).

- [ ] **Step 3: Add the iterate request model and route to `app/routes/api.py`**

Add after `GenerateRequest`:
```python
class IterateRequest(GenerateRequest):
    previous_prompt: str
```

Add after the `generate` route:
```python
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
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
pytest tests/test_api_iterate.py -v
```
Expected: PASS (all 4 tests).

- [ ] **Step 5: Run the full suite**

```bash
pytest -v
```
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add app/routes/api.py tests/test_api_iterate.py
git commit -m "Add POST /api/iterate endpoint"
```

---

## Task 13: API Route — `POST /api/from-image`

**Files:**
- Modify: `app/routes/api.py`
- Create: `tests/test_api_from_image.py`

**Interfaces:**
- Consumes: same as Task 11/12, plus `base64.b64encode` (stdlib) to build the `image_data_uri` passed to `call_openrouter`.
- Produces: adds `POST /api/from-image` (multipart form) to `app.routes.api.router`.

- [ ] **Step 1: Write the failing `/api/from-image` tests**

`tests/test_api_from_image.py`:
```python
import base64
from unittest.mock import patch

from app import families

# 1x1 transparent PNG
PNG_BYTES = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk"
    "+A8AAQUBAScY42YAAAAASUVORK5CYII="
)


def _create_family(tmp_path, has_negative_prompt=True):
    return families.create_family(
        "SDXL", "family instructions", has_negative_prompt,
        path=str(tmp_path / "families.json"),
    )


def test_from_image_requires_auth(api_client):
    response = api_client.post(
        "/api/from-image",
        files={"image": ("test.png", PNG_BYTES, "image/png")},
        data={"family_id": "x", "vision_model": "m"},
    )

    assert response.status_code == 401


@patch("app.routes.api.call_openrouter")
def test_from_image_returns_parsed_prompt(mock_call, api_client, auth_headers, tmp_path):
    family = _create_family(tmp_path)
    mock_call.return_value = "POSITIVE: a mountain landscape\nNEGATIVE: blurry"

    response = api_client.post(
        "/api/from-image",
        files={"image": ("test.png", PNG_BYTES, "image/png")},
        data={"family_id": family["id"], "vision_model": "anthropic/claude-sonnet-4"},
        auth=auth_headers,
    )

    assert response.status_code == 200
    assert response.json() == {
        "positive_prompt": "a mountain landscape",
        "negative_prompt": "blurry",
    }


@patch("app.routes.api.call_openrouter")
def test_from_image_passes_base64_data_uri(mock_call, api_client, auth_headers, tmp_path):
    family = _create_family(tmp_path)
    mock_call.return_value = "POSITIVE: ok\nNEGATIVE: "

    api_client.post(
        "/api/from-image",
        files={"image": ("test.png", PNG_BYTES, "image/png")},
        data={"family_id": family["id"], "vision_model": "m"},
        auth=auth_headers,
    )

    _, kwargs = mock_call.call_args
    assert kwargs["image_data_uri"].startswith("data:image/png;base64,")


def test_from_image_rejects_unsupported_content_type(api_client, auth_headers, tmp_path):
    family = _create_family(tmp_path)

    response = api_client.post(
        "/api/from-image",
        files={"image": ("test.gif", b"not-really-a-gif", "image/gif")},
        data={"family_id": family["id"], "vision_model": "m"},
        auth=auth_headers,
    )

    assert response.status_code == 400


def test_from_image_rejects_oversized_image(api_client, auth_headers, tmp_path):
    family = _create_family(tmp_path)
    oversized = b"0" * (10 * 1024 * 1024 + 1)

    response = api_client.post(
        "/api/from-image",
        files={"image": ("big.png", oversized, "image/png")},
        data={"family_id": family["id"], "vision_model": "m"},
        auth=auth_headers,
    )

    assert response.status_code == 400


@patch("app.routes.api.call_openrouter")
def test_from_image_appends_history_with_vision_model_and_no_llm_model(mock_call, api_client, auth_headers, tmp_path):
    family = _create_family(tmp_path)
    mock_call.return_value = "POSITIVE: ok\nNEGATIVE: "

    api_client.post(
        "/api/from-image",
        files={"image": ("test.png", PNG_BYTES, "image/png")},
        data={"family_id": family["id"], "vision_model": "anthropic/claude-sonnet-4"},
        auth=auth_headers,
    )

    from app import history
    entries = history.list_entries(path=str(tmp_path / "history.jsonl"))
    assert entries[0]["mode"] == "image"
    assert entries[0]["vision_model"] == "anthropic/claude-sonnet-4"
    assert entries[0]["llm_model"] is None
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
pytest tests/test_api_from_image.py -v
```
Expected: FAIL (`404 Not Found` for `/api/from-image`).

- [ ] **Step 3: Add the route to `app/routes/api.py`**

Add near the top, after the existing imports:
```python
import base64

from fastapi import File, Form, UploadFile
```

Add constants and the route at the end of the file:
```python
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
    system = prompts.build_system_prompt(
        system_prompt.read_system_prompt(), family, is_iteration=False
    )
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
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
pytest tests/test_api_from_image.py -v
```
Expected: PASS (all 6 tests).

- [ ] **Step 5: Run the full suite**

```bash
pytest -v
```
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add app/routes/api.py tests/test_api_from_image.py
git commit -m "Add POST /api/from-image endpoint"
```

---

## Task 14: API Routes — `GET /api/history` and `GET /api/characters`

**Files:**
- Modify: `app/routes/api.py`
- Create: `tests/test_api_misc.py`

**Interfaces:**
- Consumes: `app.history.list_entries` (Task 6), `app.characters.list_characters` (Task 5).
- Produces: adds `GET /api/history` and `GET /api/characters` to `app.routes.api.router`.

- [ ] **Step 1: Write the failing tests**

`tests/test_api_misc.py`:
```python
from unittest.mock import patch

from app import characters, families, history


def test_get_history_requires_auth(api_client):
    response = api_client.get("/api/history")

    assert response.status_code == 401


def test_get_history_returns_entries_newest_first(api_client, auth_headers, tmp_path):
    path = str(tmp_path / "history.jsonl")
    history.append_entry(
        mode="generate", family_id="f", family_name="SDXL", llm_model="m",
        vision_model=None, temperature=0.7, user_input="first", example_prompts="",
        previous_prompt=None, positive_prompt="first result", negative_prompt="",
        path=path,
    )
    history.append_entry(
        mode="generate", family_id="f", family_name="SDXL", llm_model="m",
        vision_model=None, temperature=0.7, user_input="second", example_prompts="",
        previous_prompt=None, positive_prompt="second result", negative_prompt="",
        path=path,
    )

    response = api_client.get("/api/history", auth=auth_headers)

    assert response.status_code == 200
    body = response.json()
    assert [e["positive_prompt"] for e in body] == ["second result", "first result"]


def test_get_characters_requires_auth(api_client):
    response = api_client.get("/api/characters")

    assert response.status_code == 401


def test_get_characters_returns_list(api_client, auth_headers, tmp_path):
    characters.create_character("Warrior", "a fierce warrior", path=str(tmp_path / "characters.json"))

    response = api_client.get("/api/characters", auth=auth_headers)

    assert response.status_code == 200
    assert response.json()[0]["name"] == "Warrior"
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
pytest tests/test_api_misc.py -v
```
Expected: FAIL (`404 Not Found` for both routes).

- [ ] **Step 3: Add the routes to `app/routes/api.py`**

Add `characters` to the existing `from .. import families, history, prompts, system_prompt` line, making it:
```python
from .. import characters, families, history, prompts, system_prompt
```

Add these two routes (anywhere after `router = APIRouter(prefix="/api")`):
```python
@router.get("/history")
def get_history():
    return history.list_entries()


@router.get("/characters")
def get_characters():
    return characters.list_characters()
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
pytest tests/test_api_misc.py -v
```
Expected: PASS (all 4 tests).

- [ ] **Step 5: Run the full suite**

```bash
pytest -v
```
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add app/routes/api.py tests/test_api_misc.py
git commit -m "Add GET /api/history and GET /api/characters endpoints"
```

---

## Task 15: Admin Routes — Families, Characters, System Prompt CRUD

**Files:**
- Create: `app/routes/admin.py`
- Create: `tests/test_admin.py`
- Modify: `app/main.py`

**Interfaces:**
- Consumes: `app.families` (Task 4), `app.characters` (Task 5), `app.system_prompt` (Task 7).
- Produces: `app.routes.admin.router` — a `fastapi.APIRouter` mounted at `/api/admin`, included into `app.main.app` in this task. Routes:
  - `GET/POST /api/admin/families`, `PUT/DELETE /api/admin/families/{family_id}`
  - `GET/POST /api/admin/characters`, `PUT/DELETE /api/admin/characters/{character_id}`
  - `GET/PUT /api/admin/system-prompt`

- [ ] **Step 1: Write the failing admin tests**

`tests/test_admin.py`:
```python
def test_admin_families_requires_auth(api_client):
    response = api_client.get("/api/admin/families")

    assert response.status_code == 401


def test_create_list_update_delete_family(api_client, auth_headers):
    create_response = api_client.post(
        "/api/admin/families",
        json={"name": "SDXL", "instructions": "rules", "has_negative_prompt": True},
        auth=auth_headers,
    )
    assert create_response.status_code == 200
    family_id = create_response.json()["id"]

    list_response = api_client.get("/api/admin/families", auth=auth_headers)
    assert len(list_response.json()) == 1

    update_response = api_client.put(
        f"/api/admin/families/{family_id}",
        json={"name": "SDXL v2", "instructions": "new rules", "has_negative_prompt": False},
        auth=auth_headers,
    )
    assert update_response.status_code == 200
    assert update_response.json()["name"] == "SDXL v2"

    delete_response = api_client.delete(f"/api/admin/families/{family_id}", auth=auth_headers)
    assert delete_response.status_code == 200
    assert api_client.get("/api/admin/families", auth=auth_headers).json() == []


def test_update_unknown_family_returns_404(api_client, auth_headers):
    response = api_client.put(
        "/api/admin/families/nonexistent",
        json={"name": "x", "instructions": "y", "has_negative_prompt": True},
        auth=auth_headers,
    )

    assert response.status_code == 404


def test_delete_unknown_family_returns_404(api_client, auth_headers):
    response = api_client.delete("/api/admin/families/nonexistent", auth=auth_headers)

    assert response.status_code == 404


def test_create_list_update_delete_character(api_client, auth_headers):
    create_response = api_client.post(
        "/api/admin/characters",
        json={"name": "Warrior", "text": "a fierce warrior"},
        auth=auth_headers,
    )
    assert create_response.status_code == 200
    character_id = create_response.json()["id"]

    update_response = api_client.put(
        f"/api/admin/characters/{character_id}",
        json={"name": "Warrior v2", "text": "an even fiercer warrior"},
        auth=auth_headers,
    )
    assert update_response.status_code == 200
    assert update_response.json()["name"] == "Warrior v2"

    delete_response = api_client.delete(f"/api/admin/characters/{character_id}", auth=auth_headers)
    assert delete_response.status_code == 200
    assert api_client.get("/api/admin/characters", auth=auth_headers).json() == []


def test_get_and_update_system_prompt(api_client, auth_headers):
    get_response = api_client.get("/api/admin/system-prompt", auth=auth_headers)
    assert get_response.status_code == 200
    assert get_response.json() == {"text": ""}

    put_response = api_client.put(
        "/api/admin/system-prompt", json={"text": "You are an expert."}, auth=auth_headers
    )
    assert put_response.status_code == 200

    assert api_client.get("/api/admin/system-prompt", auth=auth_headers).json() == {
        "text": "You are an expert."
    }
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
pytest tests/test_admin.py -v
```
Expected: FAIL (`ModuleNotFoundError: No module named 'app.routes.admin'`).

- [ ] **Step 3: Write `app/routes/admin.py`**

```python
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


@router.get("/system-prompt")
def get_system_prompt():
    return {"text": system_prompt.read_system_prompt()}


@router.put("/system-prompt")
def update_system_prompt(payload: SystemPromptPayload):
    system_prompt.write_system_prompt(payload.text)
    return {"text": payload.text}
```

- [ ] **Step 4: Wire the admin router into the app**

Modify `app/main.py`:
```python
from fastapi import Depends, FastAPI

from .auth import require_auth
from .routes import admin as admin_routes
from .routes import api as api_routes

app = FastAPI(title="Prompt Enhancer", dependencies=[Depends(require_auth)])

app.include_router(api_routes.router)
app.include_router(admin_routes.router)
```

- [ ] **Step 5: Run the tests to verify they pass**

```bash
pytest tests/test_admin.py -v
```
Expected: PASS (all 7 tests).

- [ ] **Step 6: Run the full suite**

```bash
pytest -v
```
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add app/routes/admin.py app/main.py tests/test_admin.py
git commit -m "Add admin CRUD endpoints for families, characters, and system prompt"
```

---

## Task 16: Page Routes and Startup Seeding

**Files:**
- Create: `app/templates/base.html`, `app/templates/index.html` (minimal placeholder body — full markup lands in Task 17), `app/templates/historial.html`, `app/templates/admin.html`
- Create: `app/static/.gitkeep`
- Create: `app/routes/pages.py`
- Modify: `app/main.py`
- Create: `tests/test_pages.py`

**Interfaces:**
- Consumes: `app.seed.ensure_seed_data` (Task 8), `app.families.list_families` (Task 4), `app.config.get_settings` (Task 2).
- Produces: `app.routes.pages.router` — a `fastapi.APIRouter` with `GET /`, `GET /historial`, `GET /admin`, included into `app.main.app` in this task; a FastAPI startup event on `app.main.app` that calls `ensure_seed_data(get_settings().data_dir)`.

This task wires page rendering end-to-end with minimal template bodies so the seeding/auth/template-loading plumbing is proven before Task 17 fills in the real UI.

- [ ] **Step 1: Write the failing page tests**

`tests/test_pages.py`:
```python
def test_index_requires_auth(api_client):
    response = api_client.get("/")

    assert response.status_code == 401


def test_index_renders_seeded_families(api_client, auth_headers):
    response = api_client.get("/", auth=auth_headers)

    assert response.status_code == 200
    assert "SDXL" in response.text
    assert "Z-Image-Turbo" in response.text


def test_historial_page_renders(api_client, auth_headers):
    response = api_client.get("/historial", auth=auth_headers)

    assert response.status_code == 200


def test_admin_page_renders(api_client, auth_headers):
    response = api_client.get("/admin", auth=auth_headers)

    assert response.status_code == 200
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
pytest tests/test_pages.py -v
```
Expected: FAIL (`404 Not Found` for `/`, `/historial`, `/admin`).

- [ ] **Step 3: Create the template files**

`app/templates/base.html`:
```html
<!doctype html>
<html lang="es">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{% block title %}Prompt Enhancer{% endblock %}</title>
  <link rel="stylesheet" href="/static/style.css">
</head>
<body>
  <nav class="topnav">
    <a href="/">Prompt Enhancer</a>
    <a href="/historial">Historial</a>
    <a href="/admin">Admin</a>
  </nav>
  <main>
    {% block content %}{% endblock %}
  </main>
  {% block scripts %}{% endblock %}
</body>
</html>
```

`app/templates/index.html` (minimal for now — replaced with the full form markup in Task 17):
```html
{% extends "base.html" %}
{% block title %}Prompt Enhancer{% endblock %}
{% block content %}
<select id="generar-family">
  {% for family in families %}
  <option value="{{ family.id }}">{{ family.name }}</option>
  {% endfor %}
</select>
{% endblock %}
```

`app/templates/historial.html`:
```html
{% extends "base.html" %}
{% block title %}Historial{% endblock %}
{% block content %}
<h1>Historial</h1>
{% endblock %}
```

`app/templates/admin.html`:
```html
{% extends "base.html" %}
{% block title %}Admin{% endblock %}
{% block content %}
<h1>Admin</h1>
{% endblock %}
```

`app/static/.gitkeep`:
```
```
(empty file, keeps the directory tracked before `style.css`/`app.js` exist)

- [ ] **Step 4: Write `app/routes/pages.py`**

```python
from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from .. import families

router = APIRouter()
templates = Jinja2Templates(directory="app/templates")


@router.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse(
        "index.html", {"request": request, "families": families.list_families()}
    )


@router.get("/historial", response_class=HTMLResponse)
def historial_page(request: Request):
    return templates.TemplateResponse("historial.html", {"request": request})


@router.get("/admin", response_class=HTMLResponse)
def admin_page(request: Request):
    return templates.TemplateResponse("admin.html", {"request": request})
```

- [ ] **Step 5: Wire the pages router and startup seeding into `app/main.py`**

```python
from fastapi import Depends, FastAPI
from fastapi.staticfiles import StaticFiles

from .auth import require_auth
from .config import get_settings
from .routes import admin as admin_routes
from .routes import api as api_routes
from .routes import pages as page_routes
from .seed import ensure_seed_data

app = FastAPI(title="Prompt Enhancer", dependencies=[Depends(require_auth)])

app.include_router(api_routes.router)
app.include_router(admin_routes.router)
app.include_router(page_routes.router)
app.mount("/static", StaticFiles(directory="app/static"), name="static")


@app.on_event("startup")
def on_startup():
    ensure_seed_data(get_settings().data_dir)
```

- [ ] **Step 6: Run the tests to verify they pass**

```bash
pytest tests/test_pages.py -v
```
Expected: PASS (all 4 tests).

- [ ] **Step 7: Run the full suite**

```bash
pytest -v
```
Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add app/templates app/static/.gitkeep app/routes/pages.py app/main.py tests/test_pages.py
git commit -m "Add page routes (index, historial, admin) with startup seeding"
```

---

## Task 17: Frontend — Generar/Iterar/Imagen Markup and Styling

**Files:**
- Modify: `app/templates/index.html`
- Create: `app/static/style.css`
- Modify: `tests/test_pages.py`

**Interfaces:**
- Consumes: the `families` template context already provided by the `/` route (Task 16).
- Produces: the full tabbed form markup that Tasks 18-19's JS will attach behavior to. Key element IDs later tasks depend on: `form-generar`, `form-iterar`, `form-imagen`, `generar-user-input`, `iterar-user-input`, `iterar-previous-prompt`, `generar-characters`, `iterar-characters`, `imagen-file`, `imagen-preview`, `result-panel`, `result-error`, `result-success`, `result-positive`, `result-negative`, `copy-positive`, `copy-negative`, `iterate-this`.

- [ ] **Step 1: Replace `app/templates/index.html` with the full markup**

```html
{% extends "base.html" %}
{% block title %}Prompt Enhancer{% endblock %}
{% block content %}
<div class="tabs">
  <button class="tab-button active" data-tab="generar" type="button">Generar</button>
  <button class="tab-button" data-tab="iterar" type="button">Iterar</button>
  <button class="tab-button" data-tab="imagen" type="button">Imagen</button>
</div>

<section id="tab-generar" class="tab-panel active">
  <form id="form-generar">
    <label for="generar-user-input">Idea</label>
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

<section id="tab-iterar" class="tab-panel">
  <form id="form-iterar">
    <label for="iterar-previous-prompt">Prompt a modificar</label>
    <textarea id="iterar-previous-prompt" name="previous_prompt" rows="4" required></textarea>

    <label for="iterar-user-input">Cambios solicitados</label>
    <textarea id="iterar-user-input" name="user_input" rows="4" required></textarea>

    <div class="characters" id="iterar-characters"></div>

    <label for="iterar-family">Familia de modelo</label>
    <select id="iterar-family" name="family_id" required>
      {% for family in families %}
      <option value="{{ family.id }}">{{ family.name }}</option>
      {% endfor %}
    </select>

    <label for="iterar-example-prompts">Prompts de ejemplo (opcional)</label>
    <textarea id="iterar-example-prompts" name="example_prompts" rows="3"></textarea>

    <label for="iterar-llm-model">Modelo LLM (OpenRouter)</label>
    <input id="iterar-llm-model" name="llm_model" type="text" value="anthropic/claude-sonnet-4" required>

    <label for="iterar-creativity">Creatividad: <span class="creativity-value">0.7</span></label>
    <input id="iterar-creativity" name="temperature" type="range" min="0" max="1" step="0.05" value="0.7">

    <button type="submit">Iterar</button>
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
    <label>Negative Prompt</label>
    <div class="copy-row">
      <textarea id="result-negative" readonly rows="2"></textarea>
      <button type="button" id="copy-negative">Copiar</button>
    </div>
    <button type="button" id="iterate-this">Iterar este prompt</button>
  </div>
</section>
{% endblock %}
{% block scripts %}
<script src="/static/app.js"></script>
{% endblock %}
```

- [ ] **Step 2: Create `app/static/style.css`**

```css
* { box-sizing: border-box; }
body { font-family: system-ui, sans-serif; margin: 0; background: #1e1e1e; color: #eee; }
.topnav { display: flex; gap: 1.5rem; padding: 1rem 1.5rem; background: #141414; }
.topnav a { color: #eee; text-decoration: none; font-weight: 600; }
.topnav a:hover { text-decoration: underline; }
main { max-width: 720px; margin: 0 auto; padding: 1.5rem; }
.tabs { display: flex; gap: 0.5rem; margin-bottom: 1rem; }
.tab-button { flex: 1; padding: 0.6rem; background: #2a2a2a; border: 1px solid #444; color: #eee; cursor: pointer; }
.tab-button.active { background: #3a6ea5; border-color: #3a6ea5; }
.tab-panel { display: none; flex-direction: column; gap: 0.75rem; }
.tab-panel.active { display: flex; }
form { display: flex; flex-direction: column; gap: 0.75rem; }
label { font-weight: 600; }
textarea, input[type="text"], select { width: 100%; padding: 0.5rem; background: #2a2a2a; border: 1px solid #444; color: #eee; }
.characters { display: flex; flex-wrap: wrap; gap: 0.4rem; }
.character-button { padding: 0.3rem 0.6rem; background: #333; border: 1px solid #555; color: #eee; cursor: pointer; font-size: 0.85rem; }
.image-preview { max-width: 100%; max-height: 200px; }
button[type="submit"], #iterate-this, .copy-row button { padding: 0.6rem; background: #3a6ea5; border: none; color: #fff; cursor: pointer; }
.result-panel { margin-top: 1.5rem; border-top: 1px solid #444; padding-top: 1rem; }
.copy-row { display: flex; gap: 0.5rem; }
.copy-row textarea { flex: 1; }
.error-banner { background: #5a1e1e; border: 1px solid #a33; color: #fff; padding: 0.75rem; border-radius: 4px; }
table { width: 100%; border-collapse: collapse; }
th, td { border-bottom: 1px solid #444; padding: 0.5rem; text-align: left; vertical-align: top; }
```

- [ ] **Step 3: Update the index page test to check for the full markup**

Add to `tests/test_pages.py` (keep the existing 4 tests, add this one):
```python
def test_index_renders_all_three_tabs(api_client, auth_headers):
    response = api_client.get("/", auth=auth_headers)

    assert response.status_code == 200
    assert 'id="form-generar"' in response.text
    assert 'id="form-iterar"' in response.text
    assert 'id="form-imagen"' in response.text
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
pytest tests/test_pages.py -v
```
Expected: PASS (all 5 tests).

- [ ] **Step 5: Commit**

```bash
git add app/templates/index.html app/static/style.css tests/test_pages.py
git commit -m "Add full Generar/Iterar/Imagen form markup and styling"
```

---

## Task 18: Frontend — Tab Switching, Generar/Iterar Submission, Result Panel, Characters

**Files:**
- Create: `app/static/app.js`

**Interfaces:**
- Consumes: `GET /api/characters`, `POST /api/generate`, `POST /api/iterate` (Tasks 11, 12, 14); the element IDs from Task 17.
- Produces: client-side behavior — no new backend interfaces. Verified manually (browser) per Task 23, since there is no JS test runner in this stack.

- [ ] **Step 1: Write `app/static/app.js`**

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

function setupIterarForm() {
  const form = document.getElementById("form-iterar");
  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    const formData = new FormData(form);
    try {
      const data = await postJSON("/api/iterate", {
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
    document.getElementById("iterar-previous-prompt").value = positive;
    document.querySelector('.tab-button[data-tab="iterar"]').click();
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
  const targets = [
    { containerId: "generar-characters", textareaId: "generar-user-input" },
    { containerId: "iterar-characters", textareaId: "iterar-user-input" },
  ];
  targets.forEach(({ containerId, textareaId }) => {
    const container = document.getElementById(containerId);
    const textarea = document.getElementById(textareaId);
    characterList.forEach((character) => {
      const button = document.createElement("button");
      button.type = "button";
      button.className = "character-button";
      button.textContent = character.name;
      button.addEventListener("click", () => insertAtCursor(textarea, character.text));
      container.appendChild(button);
    });
  });
}

document.addEventListener("DOMContentLoaded", () => {
  setupTabs();
  setupCreativitySliders();
  setupGenerarForm();
  setupIterarForm();
  setupCopyButtons();
  setupIterateHandoff();
  setupCharacterButtons();
});
```

- [ ] **Step 2: Manually smoke-test in the browser**

```bash
uvicorn app.main:app --reload
```
Open `http://localhost:8000`, log in with the Basic Auth prompt using the `.env` credentials, and verify: tabs switch, the creativity label updates while dragging the slider, and (with a valid `OPENROUTER_API_KEY` in `.env`) submitting the Generar form shows a result with working Copiar buttons. This is a manual check — automated coverage for this task lives in the API tests already passing from Tasks 11-14.

- [ ] **Step 3: Commit**

```bash
git add app/static/app.js
git commit -m "Add tab switching, Generar/Iterar submission, result panel, and character insertion"
```

---

## Task 19: Frontend — Imagen Tab Submission and Preview

**Files:**
- Modify: `app/static/app.js`

**Interfaces:**
- Consumes: `POST /api/from-image` (Task 13); the `imagen-file`, `imagen-preview`, `form-imagen` element IDs from Task 17.
- Produces: client-side behavior only, verified manually per Task 23.

- [ ] **Step 1: Add the Imagen tab handlers to `app/static/app.js`**

Add this function after `setupCharacterButtons`:
```javascript
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
```

Update the `DOMContentLoaded` listener at the bottom of the file to call it:
```javascript
document.addEventListener("DOMContentLoaded", () => {
  setupTabs();
  setupCreativitySliders();
  setupGenerarForm();
  setupIterarForm();
  setupImagenForm();
  setupCopyButtons();
  setupIterateHandoff();
  setupCharacterButtons();
});
```

- [ ] **Step 2: Manually smoke-test in the browser**

With `uvicorn app.main:app --reload` running, open the Imagen tab, choose an image file, confirm the preview appears, and (with a valid `OPENROUTER_API_KEY`) submit and confirm a result appears in the shared result panel.

- [ ] **Step 3: Commit**

```bash
git add app/static/app.js
git commit -m "Add Imagen tab preview and submission"
```

---

## Task 20: Frontend — Historial Page

**Files:**
- Modify: `app/templates/historial.html`
- Create: `app/static/historial.js`
- Modify: `tests/test_pages.py`

**Interfaces:**
- Consumes: `GET /api/history` (Task 14).
- Produces: client-side rendering only.

- [ ] **Step 1: Replace `app/templates/historial.html`**

```html
{% extends "base.html" %}
{% block title %}Historial{% endblock %}
{% block content %}
<h1>Historial</h1>
<table>
  <thead>
    <tr><th>Fecha</th><th>Modo</th><th>Familia</th><th>Positive</th><th>Negative</th></tr>
  </thead>
  <tbody id="history-body"></tbody>
</table>
{% endblock %}
{% block scripts %}
<script src="/static/historial.js"></script>
{% endblock %}
```

- [ ] **Step 2: Create `app/static/historial.js`**

```javascript
function truncate(text, length) {
  if (!text) return "";
  return text.length > length ? `${text.slice(0, length)}...` : text;
}

async function loadHistory() {
  const response = await fetch("/api/history");
  const entries = await response.json();
  const body = document.getElementById("history-body");

  entries.forEach((entry) => {
    const row = document.createElement("tr");

    const date = document.createElement("td");
    date.textContent = new Date(entry.timestamp).toLocaleString();
    row.appendChild(date);

    const mode = document.createElement("td");
    mode.textContent = entry.mode;
    row.appendChild(mode);

    const family = document.createElement("td");
    family.textContent = entry.family_name;
    row.appendChild(family);

    const positive = document.createElement("td");
    positive.textContent = truncate(entry.positive_prompt, 120);
    positive.title = entry.positive_prompt;
    row.appendChild(positive);

    const negative = document.createElement("td");
    negative.textContent = truncate(entry.negative_prompt, 80);
    negative.title = entry.negative_prompt;
    row.appendChild(negative);

    body.appendChild(row);
  });
}

document.addEventListener("DOMContentLoaded", loadHistory);
```

- [ ] **Step 3: Add a page test for the history table markup**

Add to `tests/test_pages.py`:
```python
def test_historial_page_has_history_table(api_client, auth_headers):
    response = api_client.get("/historial", auth=auth_headers)

    assert response.status_code == 200
    assert 'id="history-body"' in response.text
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
pytest tests/test_pages.py -v
```
Expected: PASS.

- [ ] **Step 5: Manually smoke-test in the browser**

Generate a prompt from the main page, then open `/historial` and confirm the entry appears with a readable timestamp.

- [ ] **Step 6: Commit**

```bash
git add app/templates/historial.html app/static/historial.js tests/test_pages.py
git commit -m "Add Historial page rendering"
```

---

## Task 21: Frontend — Admin Page

**Files:**
- Modify: `app/templates/admin.html`
- Create: `app/static/admin.js`
- Modify: `tests/test_pages.py`

**Interfaces:**
- Consumes: `/api/admin/system-prompt`, `/api/admin/families`, `/api/admin/characters` (Task 15).
- Produces: client-side rendering only.

- [ ] **Step 1: Replace `app/templates/admin.html`**

```html
{% extends "base.html" %}
{% block title %}Admin{% endblock %}
{% block content %}
<h1>Admin</h1>

<section>
  <h2>System Prompt Global</h2>
  <textarea id="system-prompt-text" rows="8"></textarea>
  <button type="button" id="save-system-prompt">Guardar</button>
</section>

<section>
  <h2>Familias</h2>
  <ul id="families-list"></ul>
  <form id="family-form">
    <input type="hidden" id="family-id">
    <label for="family-name">Nombre</label>
    <input type="text" id="family-name" required>
    <label for="family-instructions">Instrucciones</label>
    <textarea id="family-instructions" rows="6" required></textarea>
    <label><input type="checkbox" id="family-has-negative"> Usa negative prompt</label>
    <button type="submit">Guardar familia</button>
    <button type="button" id="family-cancel">Cancelar edicion</button>
  </form>
</section>

<section>
  <h2>Personajes</h2>
  <ul id="characters-list"></ul>
  <form id="character-form">
    <input type="hidden" id="character-id">
    <label for="character-name">Nombre</label>
    <input type="text" id="character-name" required>
    <label for="character-text">Texto</label>
    <textarea id="character-text" rows="3" required></textarea>
    <button type="submit">Guardar personaje</button>
    <button type="button" id="character-cancel">Cancelar edicion</button>
  </form>
</section>
{% endblock %}
{% block scripts %}
<script src="/static/admin.js"></script>
{% endblock %}
```

- [ ] **Step 2: Create `app/static/admin.js`**

```javascript
async function fetchJSON(url, options) {
  const response = await fetch(url, options);
  const data = await response.json();
  if (!response.ok) {
    throw new Error(data.detail || "Error desconocido");
  }
  return data;
}

async function loadSystemPrompt() {
  const data = await fetchJSON("/api/admin/system-prompt");
  document.getElementById("system-prompt-text").value = data.text;
}

function setupSystemPromptForm() {
  document.getElementById("save-system-prompt").addEventListener("click", async () => {
    const text = document.getElementById("system-prompt-text").value;
    await fetchJSON("/api/admin/system-prompt", {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });
  });
}

function fillFamilyForm(family) {
  document.getElementById("family-id").value = family.id;
  document.getElementById("family-name").value = family.name;
  document.getElementById("family-instructions").value = family.instructions;
  document.getElementById("family-has-negative").checked = family.has_negative_prompt;
}

function clearFamilyForm() {
  document.getElementById("family-form").reset();
  document.getElementById("family-id").value = "";
}

async function loadFamilies() {
  const items = await fetchJSON("/api/admin/families");
  const list = document.getElementById("families-list");
  list.innerHTML = "";
  items.forEach((family) => {
    const item = document.createElement("li");
    item.textContent = `${family.name} `;

    const editButton = document.createElement("button");
    editButton.type = "button";
    editButton.textContent = "Editar";
    editButton.addEventListener("click", () => fillFamilyForm(family));
    item.appendChild(editButton);

    const deleteButton = document.createElement("button");
    deleteButton.type = "button";
    deleteButton.textContent = "Eliminar";
    deleteButton.addEventListener("click", async () => {
      await fetchJSON(`/api/admin/families/${family.id}`, { method: "DELETE" });
      loadFamilies();
    });
    item.appendChild(deleteButton);

    list.appendChild(item);
  });
}

function setupFamilyForm() {
  document.getElementById("family-form").addEventListener("submit", async (event) => {
    event.preventDefault();
    const id = document.getElementById("family-id").value;
    const payload = {
      name: document.getElementById("family-name").value,
      instructions: document.getElementById("family-instructions").value,
      has_negative_prompt: document.getElementById("family-has-negative").checked,
    };
    const url = id ? `/api/admin/families/${id}` : "/api/admin/families";
    await fetchJSON(url, {
      method: id ? "PUT" : "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    clearFamilyForm();
    loadFamilies();
  });
  document.getElementById("family-cancel").addEventListener("click", clearFamilyForm);
}

function fillCharacterForm(character) {
  document.getElementById("character-id").value = character.id;
  document.getElementById("character-name").value = character.name;
  document.getElementById("character-text").value = character.text;
}

function clearCharacterForm() {
  document.getElementById("character-form").reset();
  document.getElementById("character-id").value = "";
}

async function loadCharacters() {
  const items = await fetchJSON("/api/admin/characters");
  const list = document.getElementById("characters-list");
  list.innerHTML = "";
  items.forEach((character) => {
    const item = document.createElement("li");
    item.textContent = `${character.name} `;

    const editButton = document.createElement("button");
    editButton.type = "button";
    editButton.textContent = "Editar";
    editButton.addEventListener("click", () => fillCharacterForm(character));
    item.appendChild(editButton);

    const deleteButton = document.createElement("button");
    deleteButton.type = "button";
    deleteButton.textContent = "Eliminar";
    deleteButton.addEventListener("click", async () => {
      await fetchJSON(`/api/admin/characters/${character.id}`, { method: "DELETE" });
      loadCharacters();
    });
    item.appendChild(deleteButton);

    list.appendChild(item);
  });
}

function setupCharacterForm() {
  document.getElementById("character-form").addEventListener("submit", async (event) => {
    event.preventDefault();
    const id = document.getElementById("character-id").value;
    const payload = {
      name: document.getElementById("character-name").value,
      text: document.getElementById("character-text").value,
    };
    const url = id ? `/api/admin/characters/${id}` : "/api/admin/characters";
    await fetchJSON(url, {
      method: id ? "PUT" : "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    clearCharacterForm();
    loadCharacters();
  });
  document.getElementById("character-cancel").addEventListener("click", clearCharacterForm);
}

document.addEventListener("DOMContentLoaded", () => {
  loadSystemPrompt();
  setupSystemPromptForm();
  loadFamilies();
  setupFamilyForm();
  loadCharacters();
  setupCharacterForm();
});
```

- [ ] **Step 3: Add a page test for the admin markup**

Add to `tests/test_pages.py`:
```python
def test_admin_page_has_management_sections(api_client, auth_headers):
    response = api_client.get("/admin", auth=auth_headers)

    assert response.status_code == 200
    assert 'id="family-form"' in response.text
    assert 'id="character-form"' in response.text
    assert 'id="system-prompt-text"' in response.text
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
pytest tests/test_pages.py -v
```
Expected: PASS.

- [ ] **Step 5: Manually smoke-test in the browser**

Open `/admin`, edit the global system prompt and save, then create/edit/delete a family and a character, confirming the lists refresh after each action.

- [ ] **Step 6: Commit**

```bash
git add app/templates/admin.html app/static/admin.js tests/test_pages.py
git commit -m "Add Admin page for system prompt, families, and characters management"
```

---

## Task 22: Docker Deployment and README

**Files:**
- Create: `Dockerfile`, `docker-compose.yml`
- Modify: `README.md`

**Interfaces:**
- Consumes: nothing new — packages the app built in Tasks 1-21.
- Produces: a runnable container image and compose service.

- [ ] **Step 1: Write `Dockerfile`**

```dockerfile
FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

ENV DATA_DIR=/app/data

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

- [ ] **Step 2: Write `docker-compose.yml`**

```yaml
services:
  prompt-enhancer:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./data:/app/data
    env_file:
      - .env
    restart: unless-stopped
```

- [ ] **Step 3: Rewrite `README.md`**

```markdown
# Prompt Enhancer

A small, self-hosted web app that uses an LLM (via OpenRouter) to craft image-generation prompts for SDXL, Z-Image-Turbo, and any other model family you configure.

## Features

- **Generar** — turn a natural-language idea into a ready-to-copy positive/negative prompt pair for the selected model family.
- **Iterar** — refine an existing prompt (generated here or pasted from elsewhere) with follow-up instructions.
- **Imagen** — upload an image and generate a prompt from it using an OpenRouter vision model.
- **Personajes** — reusable text snippets you can insert into the idea/changes field with one click.
- **Admin** — edit the global system prompt, manage model families (their rules and whether they use a negative prompt), and manage personajes, all from the browser.
- **Historial** — read-only log of every prompt generated.

## Setup

1. Copy `.env.example` to `.env` and fill in:
   - `OPENROUTER_API_KEY` — from https://openrouter.ai/keys
   - `ADMIN_USERNAME` / `ADMIN_PASSWORD` — credentials for the HTTP Basic Auth prompt protecting the whole app
2. `docker compose up --build`
3. Open `http://localhost:8000` and log in with the credentials above.

On first run, the app seeds `data/families.json` with SDXL and Z-Image-Turbo. Add more families (Krea 2, Pony, Flux, ...) from the Admin panel.

## Development

```
pip install -e ".[dev]"
pytest
uvicorn app.main:app --reload
```

Data (families, characters, the global system prompt, and history) lives under `data/`, which is gitignored and mounted as a Docker volume so it survives container rebuilds.
```

- [ ] **Step 4: Build the image to confirm it compiles**

```bash
docker compose build
```
Expected: image builds successfully with no errors.

- [ ] **Step 5: Commit**

```bash
git add Dockerfile docker-compose.yml README.md
git commit -m "Add Docker deployment and rewrite README for the web app"
```

---

## Task 23: End-to-End Manual Verification

**Files:** none (verification only).

**Interfaces:** exercises the full stack built in Tasks 1-22.

- [ ] **Step 1: Run the full automated test suite one last time**

```bash
pytest -v
```
Expected: all tests PASS.

- [ ] **Step 2: Start the app via Docker Compose**

```bash
cp .env.example .env
# edit .env: set a real OPENROUTER_API_KEY and a chosen ADMIN_PASSWORD
docker compose up --build
```

- [ ] **Step 3: Verify the Generar flow**

Open `http://localhost:8000`, log in, go to the Generar tab, select SDXL, enter an idea, click a character button to confirm it inserts text, submit, and confirm a positive/negative prompt appears with working Copiar buttons.

- [ ] **Step 4: Verify the Iterar flow**

From the Generar result, click "Iterar este prompt", confirm the Iterar tab opens with the previous prompt pre-filled, enter a change instruction, submit, and confirm an updated prompt appears.

- [ ] **Step 5: Verify the Imagen flow**

Go to the Imagen tab, upload a JPEG/PNG, confirm the preview renders, select a vision-capable model, submit, and confirm a prompt appears.

- [ ] **Step 6: Verify Historial**

Open `/historial` and confirm the three generations above appear, newest first, with truncated/expandable prompts.

- [ ] **Step 7: Verify Admin**

Open `/admin`: edit the global system prompt and save; add a new family (e.g. "Flux") with `has_negative_prompt` unchecked and confirm it appears in the Generar family selector after reloading `/`; add, edit, and delete a character, confirming it appears/disappears from the Generar/Iterar character buttons after reloading `/`.

- [ ] **Step 8: Verify auth**

Open the app in a private/incognito window and confirm every page (`/`, `/historial`, `/admin`) and API route prompts for credentials, and that wrong credentials are rejected.

- [ ] **Step 9: Verify persistence across restarts**

```bash
docker compose down
docker compose up
```
Confirm the families/characters/system prompt/history created above are still present after the restart (proving the `./data` volume mount works).

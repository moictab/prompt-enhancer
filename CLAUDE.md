# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```
pip install -e ".[dev]"          # install app + dev deps (pytest, httpx) from pyproject.toml
pytest                           # run the full test suite
pytest tests/test_prompts.py     # run a single test file
pytest tests/test_prompts.py::test_parse_response_splits_positive_and_negative  # run a single test
uvicorn app.main:app --reload    # run the dev server at http://localhost:8000
docker compose up --build        # build and run the containerized app (reads .env, mounts ./data)
```

Copy `.env.example` to `.env` before running anything — `OPENROUTER_API_KEY`, `ADMIN_USERNAME`, `ADMIN_PASSWORD`. The app refuses to start (raises in `lifespan`) if `ADMIN_PASSWORD` is unset, so tests set all three via `monkeypatch.setenv` in `tests/conftest.py`'s `api_client` fixture rather than relying on a `.env` file.

## Architecture

FastAPI + Jinja2 + vanilla JS. No build step (no bundler, no npm), no database — every persistent thing is a flat JSON or JSONL file under `data/` (gitignored, Docker-volume-mounted). `app/storage.py` is the only place that touches the filesystem for JSON: `read_json`/`write_json` (atomic via tempfile + `os.replace`) for families/characters, `append_jsonl`/`read_jsonl` for history. Every domain module (`families.py`, `characters.py`, `history.py`, `system_prompt.py`) follows the same shape: a `_default_path()` derived from `get_settings().data_dir`, plain functions that accept an optional `path` override (this is how tests get isolation — `tests/conftest.py` sets `DATA_DIR` to `tmp_path` per test rather than every module needing a path argument at the call site), and no classes/ORM — just lists/dicts of plain data read-modify-written wholesale on every mutation.

**Request flow.** `app/main.py` wires a single FastAPI app with a global `Depends(require_auth)` dependency, so *every* route (including static-adjacent page routes) is behind HTTP Basic Auth — there's no per-route auth annotation to forget. `app/auth.py` compares credentials with `secrets.compare_digest` against `Settings.admin_username/admin_password`. On startup, `lifespan()` in `main.py` calls `seed.ensure_seed_data(data_dir)`, which creates `families.json` (seeded with SDXL and Z-Image-Turbo, each with hardcoded `*_INSTRUCTIONS` text), an empty `characters.json`, and `system_prompts.json` (seeded with three independent default prompts, one per mode: `generate`, `iterate`, `image`) — but only if those files don't already exist, so it's a one-time bootstrap, not a sync.

**`routes/api.py`** is where generation actually happens, and there are two endpoints, not three: `POST /api/generate` handles both the "generate" and "iterate" modes (an optional `previous_prompt` field decides which — blank means generate-from-scratch, non-blank means iterate-on-that-prompt), and `POST /api/from-image` handles the vision mode. All three *modes* (`generate`/`iterate`/`image`) follow the same composition: look up the family via `families.get_family(family_id)` (404 if unknown) → build a system prompt with `prompts.build_system_prompt(mode, family)`, which internally reads that mode's independent prompt from `data/system_prompts.json` and appends the family's instructions → build a user message with `prompts.build_user_message(mode, ...)` → call `openrouter_client.call_openrouter(...)` with the appropriate model (`llm_model` for generate/iterate, `vision_model` + `image_data_uri` for from-image) → parse the raw `POSITIVE:`/`NEGATIVE:` response via `prompts.parse_response(response, family["has_negative_prompt"])` → log everything via `history.append_entry(...)` → return `{positive_prompt, negative_prompt}`. Unlike the other two modes, `generate` and `iterate` share a single route and Pydantic model — the route computes `mode` from `previous_prompt` before doing anything else, and everything downstream (system prompt, user message, history entry) is driven by that one string.

`has_negative_prompt` (set per-family in `families.json`/admin UI) is the single flag that drives negative-prompt suppression: `prompts.parse_response` always parses out whatever the LLM put after `NEGATIVE:`, but forces it to `""` when the family says it doesn't use one (e.g. Z-Image-Turbo, whose instructions also tell the LLM not to bother generating one — the code-level suppression is the actual guarantee, not the prompt wording). The frontend mirrors this by hiding the negative-prompt group in the result panel when it's empty.

**`routes/admin.py`** is a thin CRUD layer with no logic of its own — every endpoint is a one-line pass-through to `families`/`characters`/`system_prompt` functions, translating `None`/`False` returns from those modules into 404s. `routes/pages.py` only renders templates (index needs the families list for the `<select>` dropdowns; historial and admin pages fetch their own data client-side via JS).

**Frontend.** One Jinja2 template + one vanilla JS file per page, no shared frontend framework or build step: `index.html`/`app.js` (two tabs — Generar (which itself covers both generate and iterate, via the "existing prompt" field) and Imagen — toggled by CSS class, each a plain form posting JSON or FormData to `/api/*`; "personajes" render as buttons that insert their text at the textarea cursor; a "Iterar este prompt" button fills the "existing prompt" field in the same Generar tab with the just-generated positive prompt), `admin.html`/`admin.js` (three independent system prompt textareas, one per mode, + family/character list-and-form CRUD panels, all calling `/api/admin/*`), `historial.html`/`historial.js` (fetches `/api/history`, which returns entries newest-first, and renders a table with truncated/title-attribute prompt text). All three JS files share the same hand-rolled `fetch`-wraps-JSON-with-error-on-non-2xx pattern (duplicated per file, not factored into a shared module).

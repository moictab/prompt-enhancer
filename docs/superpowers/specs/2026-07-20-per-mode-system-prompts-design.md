# Per-Mode System Prompts + Generar/Iterar Merge — Design

**Date:** 2026-07-20
**Status:** Approved for planning

## Context

The app currently has a single global system prompt (`data/system_prompt.txt`, admin-editable) shared across all three generation modes, plus a hardcoded `ITERATION_ADDENDUM` constant in `prompts.py` appended only in iterate mode. The user wants three fully independent, admin-editable global prompts — one per mode — since analyzing an image, generating from scratch, and refining an existing prompt are different-enough tasks that a shared base doesn't fit well. Separately, since the backend already collapses "generate vs. iterate" into a single boolean (`is_iteration`) based on whether a previous prompt was supplied, the user asked to merge the Generar and Iterar UI tabs and their two backend endpoints into one, to avoid the growing route duplication already flagged in code review.

## Goals

1. Three independent, admin-editable global system prompts: `generate`, `iterate`, `image`.
2. Merge `POST /api/generate` and `POST /api/iterate` into a single endpoint (`POST /api/generate`), with `previous_prompt` becoming an optional field on the same request model.
3. Merge the Generar and Iterar tabs into a single "Generar" tab: an "existing prompt" textarea that, when empty, means generate-from-scratch, and when filled, means iterate-on-that-prompt.
4. Admin panel manages the three prompts as three independent textareas/save actions instead of one.

## Non-Goals

- No change to the Imagen tab/flow beyond reading its system prompt from the new `image` key instead of the old shared file.
- No migration of any admin-edited customization from the old `system_prompt.txt` — no prompt has been customized via the admin panel yet on any deployment, so the old file is simply superseded and left orphaned on disk.
- No change to `history.jsonl`'s shape — `mode` still logs `"generate"`, `"iterate"`, or `"image"` exactly as today; only which route decides that value changes.

## Data Model

**`data/system_prompts.json`** replaces `data/system_prompt.txt`:
```json
{ "generate": "...", "iterate": "...", "image": "..." }
```
`app/system_prompt.py` changes from a single read/write pair to:
- `read_system_prompt(mode: str, path: str | None = None) -> str` — reads the whole JSON dict via `app.storage.read_json`, returns `data.get(mode, "")`.
- `write_system_prompt(mode: str, text: str, path: str | None = None) -> None` — reads the whole dict (or `{}` if absent), sets `data[mode] = text`, writes it back via `app.storage.write_json`.

`mode` is always one of the literal strings `"generate"`, `"iterate"`, `"image"` — the same three values already used by `prompts.build_user_message`'s `mode` parameter and by `history.append_entry`'s `mode` field. No new mode vocabulary is introduced.

**Seed data** (`app/seed.py`): `ensure_seed_data` seeds all three keys (still only if `system_prompts.json` doesn't already exist — same idempotency guarantee as today).
- `generate`: the current `DEFAULT_GLOBAL_SYSTEM_PROMPT` content, trimmed to focus only on transforming a natural-language idea into a prompt from scratch (drop the "or requested changes... or an attached image" hedging, since that's no longer this prompt's job).
- `iterate`: the current `ITERATION_ADDENDUM` content, promoted into a full standalone prompt — role + output format contract + the preserve/apply-changes/maintain-style rules that `ITERATION_ADDENDUM` already states.
- `image`: new content — role + output format contract + instructions to analyze the attached image's subject/composition/lighting/style and translate it into a prompt following the family's rules, prioritizing any accompanying user text for adjustments.

`ITERATION_ADDENDUM` is removed as a constant from `prompts.py` — its content now lives entirely in `system_prompts.json["iterate"]`, editable from the admin panel.

## Backend

**`prompts.build_system_prompt(mode: str, family: dict) -> str`** (signature change — drops `is_iteration`, adds `mode`):
```
system_prompt.read_system_prompt(mode) + "\n\n" + family["instructions"]
```
Simpler than today: no addendum concatenation, no boolean flag — the mode string alone determines which global prompt is used.

**`app/routes/api.py`**:
- `GenerateRequest` gains `previous_prompt: str = ""` (optional; empty string, not `None`, to match the existing "blank means absent" convention already used for `example_prompts`).
- `IterateRequest` is deleted.
- `POST /api/iterate` is deleted.
- `POST /api/generate` becomes the single entry point for both cases:
  1. Validate `user_input` non-blank (unchanged). `previous_prompt` needs no required-field validation — it's optional by design, and there is no "blank previous_prompt" error case anymore (blank simply means generate-from-scratch, not a validation failure).
  2. `is_iteration = bool(req.previous_prompt.strip())`.
  3. `mode = "iterate" if is_iteration else "generate"`.
  4. `family = families.get_family(req.family_id)` (404 if unknown, unchanged).
  5. `system = prompts.build_system_prompt(mode, family)`.
  6. `user_message = prompts.build_user_message(mode, req.user_input, previous_prompt=req.previous_prompt or None, example_prompts=req.example_prompts)`.
  7. Call `openrouter_client.call_openrouter(...)` (unchanged, still 502 on `RuntimeError`).
  8. `positive, negative = prompts.parse_response(response, family["has_negative_prompt"])` (unchanged).
  9. `history.append_entry(mode=mode, ..., previous_prompt=req.previous_prompt or None, ...)` (unchanged shape; `mode` and `previous_prompt` are simply computed by this one route now instead of two).

**`app/routes/admin.py`**: the single `GET`/`PUT /api/admin/system-prompt` pair becomes:
- `GET /api/admin/system-prompt/{mode}` → `{"text": system_prompt.read_system_prompt(mode)}`, `400` if `mode` isn't one of `generate`/`iterate`/`image`.
- `PUT /api/admin/system-prompt/{mode}` → same validation, then `system_prompt.write_system_prompt(mode, payload.text)`.

## Frontend

- **Tabs**: Generar (merged) and Imagen — two tabs instead of three. The Iterar tab and its standalone form are removed entirely.
- **Generar tab** gains one field: a "Prompt existente (déjalo vacío para generar desde cero)" textarea, positioned where the old Iterar tab's `previous_prompt` field was conceptually. All other fields (idea/changes text, family select, example prompts, `llm_model`, creativity slider, character buttons) are shared, matching what both tabs already had.
- `app.js`: `setupIterarForm` is deleted; `setupGenerarForm` sends `previous_prompt` alongside the existing fields to `/api/generate`. The "Iterar este prompt" button no longer switches tabs (there's only one tab to switch to) — it just writes the `positive_prompt` into the "existing prompt" textarea, already in view.
- **Admin page**: "System Prompt Global" section becomes three independent textarea + save-button pairs (Generar / Iterar / Imagen), each calling `GET`/`PUT /api/admin/system-prompt/{mode}` with its own `mode`.

## Testing

- `tests/test_system_prompt.py` updated for the `mode`-keyed read/write signatures.
- `tests/test_seed.py` updated to assert all three keys exist in `system_prompts.json` after seeding.
- `tests/test_prompts.py` updated: `build_system_prompt` tests now pass `mode` instead of `is_iteration`; the iteration-addendum-specific assertions move to checking that `mode="iterate"` pulls the `iterate` prompt (via a stub/fixture), not that a constant gets appended.
- `tests/test_api_iterate.py` is deleted; its four test cases (auth-required, happy-path, blank-previous-prompt-rejected — **dropped**, since blank is no longer an error — and history-shape) are folded into `tests/test_api_generate.py` as the "with `previous_prompt`" cases alongside the existing "without" cases.
- `tests/test_admin.py` updated for the `{mode}`-parameterized system-prompt routes (three round-trips instead of one, plus an invalid-mode-returns-400 case).
- `tests/test_pages.py`'s admin-page-markup test updated to check for three system-prompt textareas instead of one.
